"""PV power forecast using a clear-sky solar model.

Solar constant: 1361 W/m² (Kopp & Lean 2011).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from eie.forecasting.horizon import ForecastStep, TimeHorizon


@dataclass(frozen=True)
class PVForecastPoint:
    step: ForecastStep
    irradiance_w_m2: float
    pv_power_w: float
    confidence: float


@dataclass
class PVForecast:
    panel_area_m2: float
    panel_efficiency: float
    cloud_degradation_factor: float = 1.0
    latitude_deg: float = 51.5
    solar_constant_w_m2: float = 1361.0

    def __post_init__(self) -> None:
        if self.panel_area_m2 <= 0:
            raise ValueError("panel_area_m2 must be > 0")
        if not (0.0 < self.panel_efficiency <= 1.0):
            raise ValueError("panel_efficiency must be in (0, 1]")
        if not (0.0 <= self.cloud_degradation_factor <= 1.0):
            raise ValueError("cloud_degradation_factor must be in [0, 1]")

    def _clear_sky_irradiance(self, dt_utc: "datetime") -> float:  # type: ignore[name-defined]
        day_of_year = dt_utc.timetuple().tm_yday
        b = 2.0 * math.pi * (day_of_year - 1) / 365.0
        declination_deg = (0.006918 - 0.399912*math.cos(b) + 0.070257*math.sin(b)
            - 0.006758*math.cos(2*b) + 0.000907*math.sin(2*b)
            - 0.002697*math.cos(3*b) + 0.00148*math.sin(3*b)) * (180.0/math.pi)
        lat_rad = math.radians(self.latitude_deg)
        dec_rad = math.radians(declination_deg)
        solar_hour = dt_utc.hour + dt_utc.minute/60.0 + dt_utc.second/3600.0
        ha_rad = math.radians((solar_hour - 12.0) * 15.0)
        cos_zenith = math.sin(lat_rad)*math.sin(dec_rad) + math.cos(lat_rad)*math.cos(dec_rad)*math.cos(ha_rad)
        return max(0.0, self.solar_constant_w_m2 * cos_zenith) * self.cloud_degradation_factor

    def forecast(self, horizon: TimeHorizon) -> list[PVForecastPoint]:
        points = []
        for step in horizon.steps:
            irr = self._clear_sky_irradiance(step.midpoint)
            points.append(PVForecastPoint(step=step, irradiance_w_m2=irr,
                pv_power_w=irr * self.panel_area_m2 * self.panel_efficiency,
                confidence=max(0.3, 1.0 - 0.02 * step.step_index)))
        return points
