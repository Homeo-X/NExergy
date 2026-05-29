"""Tests for forecasting: TimeHorizon, PVForecast, LoadForecast, ForecastBundle."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from eie.forecasting.bundle import ForecastBundle, make_bundle
from eie.forecasting.horizon import ForecastStep, TimeHorizon
from eie.forecasting.load import LoadForecast, LoadForecastPoint
from eie.forecasting.solar import PVForecast, PVForecastPoint


@pytest.fixture
def noon_utc() -> datetime:
    return datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)

@pytest.fixture
def midnight_utc() -> datetime:
    return datetime(2026, 6, 21, 0, 0, tzinfo=timezone.utc)

@pytest.fixture
def horizon_24h(noon_utc: datetime) -> TimeHorizon:
    return TimeHorizon(start=noon_utc, step_duration_s=3600.0, n_steps=24)

@pytest.fixture
def horizon_short(noon_utc: datetime) -> TimeHorizon:
    return TimeHorizon(start=noon_utc, step_duration_s=900.0, n_steps=4)

def test_horizon_step_count(horizon_24h: TimeHorizon) -> None:
    assert len(horizon_24h.steps) == 24

def test_horizon_end(horizon_24h: TimeHorizon, noon_utc: datetime) -> None:
    assert horizon_24h.end == noon_utc + timedelta(hours=24)

def test_horizon_total_duration(horizon_24h: TimeHorizon) -> None:
    assert horizon_24h.total_duration_s == pytest.approx(24 * 3600.0)

def test_horizon_step_at(horizon_24h: TimeHorizon) -> None:
    assert horizon_24h.step_at(0).step_index == 0
    assert horizon_24h.step_at(23).step_index == 23

def test_horizon_step_at_out_of_range(horizon_24h: TimeHorizon) -> None:
    with pytest.raises(IndexError): horizon_24h.step_at(24)
    with pytest.raises(IndexError): horizon_24h.step_at(-1)

def test_horizon_step_continuity(horizon_24h: TimeHorizon) -> None:
    for i in range(len(horizon_24h.steps) - 1):
        assert horizon_24h.steps[i].end == horizon_24h.steps[i + 1].start

def test_horizon_invalid_step_duration() -> None:
    with pytest.raises(ValueError):
        TimeHorizon(start=datetime(2026, 1, 1, tzinfo=timezone.utc), step_duration_s=0.0, n_steps=10)

def test_horizon_invalid_n_steps() -> None:
    with pytest.raises(ValueError):
        TimeHorizon(start=datetime(2026, 1, 1, tzinfo=timezone.utc), step_duration_s=3600.0, n_steps=0)

def test_forecast_step_midpoint(horizon_short: TimeHorizon) -> None:
    step = horizon_short.steps[0]
    assert step.midpoint == step.start + timedelta(seconds=450.0)

def test_forecast_step_invalid_order() -> None:
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        ForecastStep(step_index=0, start=t, end=t, duration_s=900.0)

@pytest.fixture
def pv_forecast() -> PVForecast:
    return PVForecast(panel_area_m2=20.0, panel_efficiency=0.20)

def test_pv_forecast_length(pv_forecast: PVForecast, horizon_24h: TimeHorizon) -> None:
    assert len(pv_forecast.forecast(horizon_24h)) == 24

def test_pv_noon_positive(pv_forecast: PVForecast, noon_utc: datetime) -> None:
    points = pv_forecast.forecast(TimeHorizon(start=noon_utc, step_duration_s=3600.0, n_steps=1))
    assert points[0].pv_power_w > 0 and points[0].irradiance_w_m2 > 0

def test_pv_midnight_zero(pv_forecast: PVForecast, midnight_utc: datetime) -> None:
    points = pv_forecast.forecast(TimeHorizon(start=midnight_utc, step_duration_s=3600.0, n_steps=1))
    assert points[0].pv_power_w == pytest.approx(0.0) and points[0].irradiance_w_m2 == pytest.approx(0.0)

def test_pv_cloud_degradation(noon_utc: datetime) -> None:
    horizon = TimeHorizon(start=noon_utc, step_duration_s=3600.0, n_steps=1)
    clear = PVForecast(panel_area_m2=10.0, panel_efficiency=0.20, cloud_degradation_factor=1.0).forecast(horizon)
    cloudy = PVForecast(panel_area_m2=10.0, panel_efficiency=0.20, cloud_degradation_factor=0.4).forecast(horizon)
    assert cloudy[0].pv_power_w < clear[0].pv_power_w

def test_pv_forecast_confidence_decreases(pv_forecast: PVForecast, horizon_24h: TimeHorizon) -> None:
    points = pv_forecast.forecast(horizon_24h)
    assert points[0].confidence >= points[-1].confidence

def test_pv_forecast_invalid_efficiency() -> None:
    with pytest.raises(ValueError): PVForecast(panel_area_m2=10.0, panel_efficiency=1.5)

def test_pv_forecast_invalid_area() -> None:
    with pytest.raises(ValueError): PVForecast(panel_area_m2=-1.0, panel_efficiency=0.2)

def test_pv_forecast_invalid_cloud() -> None:
    with pytest.raises(ValueError): PVForecast(panel_area_m2=10.0, panel_efficiency=0.2, cloud_degradation_factor=1.5)

@pytest.fixture
def load_forecast() -> LoadForecast:
    return LoadForecast(base_thermal_w=5000.0, base_electrical_w=2000.0)

def test_load_forecast_length(load_forecast: LoadForecast, horizon_24h: TimeHorizon) -> None:
    assert len(load_forecast.forecast(horizon_24h)) == 24

def test_load_forecast_non_negative(load_forecast: LoadForecast, horizon_24h: TimeHorizon) -> None:
    for p in load_forecast.forecast(horizon_24h):
        assert p.thermal_load_w >= 0.0 and p.electrical_base_load_w >= 0.0 and p.noncritical_electrical_w >= 0.0

def test_load_noncritical_fraction(load_forecast: LoadForecast, noon_utc: datetime) -> None:
    points = load_forecast.forecast(TimeHorizon(start=noon_utc, step_duration_s=3600.0, n_steps=1))
    assert points[0].noncritical_electrical_w / points[0].electrical_base_load_w == pytest.approx(0.3, abs=1e-6)

def test_load_temperature_correction(horizon_24h: TimeHorizon) -> None:
    lf = LoadForecast(base_thermal_w=5000.0, base_electrical_w=2000.0, temperature_sensitivity_w_per_k=100.0, reference_outdoor_temp_k=283.15)
    cold_points = lf.forecast(horizon_24h, outdoor_temp_k=273.15)
    ref_points = lf.forecast(horizon_24h, outdoor_temp_k=283.15)
    for cold, ref in zip(cold_points, ref_points):
        assert cold.thermal_load_w >= ref.thermal_load_w

def test_load_forecast_invalid_profile() -> None:
    with pytest.raises(ValueError):
        LoadForecast(base_thermal_w=1000.0, base_electrical_w=1000.0, hourly_thermal_profile=[1.0]*12)

def test_load_forecast_invalid_noncritical() -> None:
    with pytest.raises(ValueError):
        LoadForecast(base_thermal_w=1000.0, base_electrical_w=1000.0, noncritical_fraction=1.5)

@pytest.fixture
def bundle(horizon_24h: TimeHorizon, pv_forecast: PVForecast, load_forecast: LoadForecast) -> ForecastBundle:
    return make_bundle(horizon_24h, pv_forecast, load_forecast, reference_temperature_k=293.15)

def test_bundle_step_count(bundle: ForecastBundle, horizon_24h: TimeHorizon) -> None:
    assert len(bundle.pv_points) == horizon_24h.n_steps and len(bundle.load_points) == horizon_24h.n_steps

def test_bundle_pv_at(bundle: ForecastBundle) -> None:
    assert isinstance(bundle.pv_at(0), PVForecastPoint)

def test_bundle_load_at(bundle: ForecastBundle) -> None:
    assert isinstance(bundle.load_at(0), LoadForecastPoint)

def test_bundle_pv_at_out_of_range(bundle: ForecastBundle) -> None:
    with pytest.raises(IndexError): bundle.pv_at(100)

def test_bundle_total_pv_energy(bundle: ForecastBundle) -> None:
    assert bundle.total_pv_energy_j > 0.0

def test_bundle_total_thermal_demand(bundle: ForecastBundle) -> None:
    assert bundle.total_thermal_demand_j > 0.0

def test_bundle_total_electrical_demand(bundle: ForecastBundle) -> None:
    assert bundle.total_electrical_demand_j > 0.0

def test_bundle_peak_pv(bundle: ForecastBundle) -> None:
    assert bundle.peak_pv_power_w >= 0.0

def test_bundle_is_frozen(bundle: ForecastBundle) -> None:
    with pytest.raises((AttributeError, TypeError)):
        bundle.reference_temperature_k = 300.0  # type: ignore[misc]

def test_bundle_length_mismatch(horizon_24h: TimeHorizon) -> None:
    pv = PVForecast(panel_area_m2=10.0, panel_efficiency=0.2)
    lf = LoadForecast(base_thermal_w=1000.0, base_electrical_w=500.0)
    with pytest.raises(ValueError, match="length"):
        ForecastBundle(bundle_id="b1", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            horizon=horizon_24h, pv_points=tuple(pv.forecast(horizon_24h))[:5],
            load_points=tuple(lf.forecast(horizon_24h)), reference_temperature_k=293.15)
