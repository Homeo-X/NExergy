"""Forecast bundle — combines PV and load forecasts for a shared horizon."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from eie.forecasting.horizon import TimeHorizon
from eie.forecasting.load import LoadForecast, LoadForecastPoint
from eie.forecasting.solar import PVForecast, PVForecastPoint


@dataclass(frozen=True)
class ForecastBundle:
    """Immutable collection of PV and load forecasts for one time horizon."""

    bundle_id: str
    created_at: datetime
    horizon: TimeHorizon
    pv_points: tuple[PVForecastPoint, ...]
    load_points: tuple[LoadForecastPoint, ...]
    reference_temperature_k: float

    def __post_init__(self) -> None:
        if len(self.pv_points) != self.horizon.n_steps:
            raise ValueError(
                f"pv_points length {len(self.pv_points)} != horizon.n_steps {self.horizon.n_steps}"
            )
        if len(self.load_points) != self.horizon.n_steps:
            raise ValueError(
                f"load_points length {len(self.load_points)} != horizon.n_steps {self.horizon.n_steps}"
            )
        if self.reference_temperature_k <= 0:
            raise ValueError("reference_temperature_k must be > 0")

    def pv_at(self, step_index: int) -> PVForecastPoint:
        if step_index < 0 or step_index >= self.horizon.n_steps:
            raise IndexError(f"step_index {step_index} out of range")
        return self.pv_points[step_index]

    def load_at(self, step_index: int) -> LoadForecastPoint:
        if step_index < 0 or step_index >= self.horizon.n_steps:
            raise IndexError(f"step_index {step_index} out of range")
        return self.load_points[step_index]

    @property
    def total_pv_energy_j(self) -> float:
        return sum(p.pv_power_w * p.step.duration_s for p in self.pv_points)

    @property
    def total_thermal_demand_j(self) -> float:
        return sum(p.thermal_load_w * p.step.duration_s for p in self.load_points)

    @property
    def total_electrical_demand_j(self) -> float:
        return sum(p.electrical_base_load_w * p.step.duration_s for p in self.load_points)

    @property
    def peak_pv_power_w(self) -> float:
        return max((p.pv_power_w for p in self.pv_points), default=0.0)

    @property
    def peak_thermal_load_w(self) -> float:
        return max((p.thermal_load_w for p in self.load_points), default=0.0)


def make_bundle(
    horizon: TimeHorizon,
    pv_forecast: PVForecast,
    load_forecast: LoadForecast,
    reference_temperature_k: float,
    outdoor_temp_k: float | None = None,
    *,
    created_at: datetime | None = None,
) -> ForecastBundle:
    """Convenience factory: compute forecasts and assemble a ForecastBundle."""
    pv_points = pv_forecast.forecast(horizon)
    load_points = load_forecast.forecast(horizon, outdoor_temp_k=outdoor_temp_k)
    return ForecastBundle(
        bundle_id=str(uuid.uuid4()),
        created_at=created_at or datetime.now(timezone.utc),
        horizon=horizon,
        pv_points=tuple(pv_points),
        load_points=tuple(load_points),
        reference_temperature_k=reference_temperature_k,
    )
