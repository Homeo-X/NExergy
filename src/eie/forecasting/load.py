"""Building load forecasting using hourly profiles with temperature correction."""

from __future__ import annotations

from dataclasses import dataclass, field

from eie.forecasting.horizon import ForecastStep, TimeHorizon


_DEFAULT_THERMAL_PROFILE = [1.4,1.3,1.3,1.2,1.2,1.3,1.5,1.6,1.4,1.2,1.0,0.9,0.9,0.9,0.9,1.0,1.1,1.3,1.5,1.6,1.5,1.4,1.4,1.4]
_DEFAULT_ELECTRICAL_PROFILE = [0.6,0.5,0.5,0.5,0.5,0.6,0.8,1.1,1.2,1.1,1.0,1.0,1.0,0.9,0.9,1.0,1.1,1.2,1.3,1.4,1.3,1.1,0.9,0.7]


@dataclass(frozen=True)
class LoadForecastPoint:
    step: ForecastStep
    thermal_load_w: float
    electrical_base_load_w: float
    noncritical_electrical_w: float
    confidence: float


@dataclass
class LoadForecast:
    base_thermal_w: float
    base_electrical_w: float
    noncritical_fraction: float = 0.3
    hourly_thermal_profile: list[float] = field(default_factory=lambda: list(_DEFAULT_THERMAL_PROFILE))
    hourly_electrical_profile: list[float] = field(default_factory=lambda: list(_DEFAULT_ELECTRICAL_PROFILE))
    temperature_sensitivity_w_per_k: float = 50.0
    reference_outdoor_temp_k: float = 283.15

    def __post_init__(self) -> None:
        if self.base_thermal_w < 0:
            raise ValueError("base_thermal_w must be >= 0")
        if self.base_electrical_w < 0:
            raise ValueError("base_electrical_w must be >= 0")
        if not (0.0 <= self.noncritical_fraction <= 1.0):
            raise ValueError("noncritical_fraction must be in [0, 1]")
        if len(self.hourly_thermal_profile) != 24:
            raise ValueError("hourly_thermal_profile must have exactly 24 entries")
        if len(self.hourly_electrical_profile) != 24:
            raise ValueError("hourly_electrical_profile must have exactly 24 entries")

    def forecast(self, horizon: TimeHorizon, outdoor_temp_k: float | None = None) -> list[LoadForecastPoint]:
        temp_k = outdoor_temp_k if outdoor_temp_k is not None else self.reference_outdoor_temp_k
        temp_correction_w = max(0.0, self.temperature_sensitivity_w_per_k * (self.reference_outdoor_temp_k - temp_k))
        points = []
        for step in horizon.steps:
            hour = step.midpoint.hour
            thermal_w = max(0.0, self.base_thermal_w * self.hourly_thermal_profile[hour] + temp_correction_w)
            electrical_w = max(0.0, self.base_electrical_w * self.hourly_electrical_profile[hour])
            points.append(LoadForecastPoint(step=step, thermal_load_w=thermal_w,
                electrical_base_load_w=electrical_w, noncritical_electrical_w=electrical_w * self.noncritical_fraction,
                confidence=max(0.3, 1.0 - 0.015 * step.step_index)))
        return points
