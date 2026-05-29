"""Tests for MPCScheduler — rolling horizon schedule generation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from eie.boundary.boundary import Boundary
from eie.control.gate import SafetyGate
from eie.control.policy import BatteryPolicy, ThermalStoragePolicy
from eie.core.enums import BoundaryType, MeasurementMethod
from eie.flows.base import Metadata
from eie.flows.storage import BatteryState, ThermalLayer, ThermalStorageState
from eie.forecasting.bundle import make_bundle
from eie.forecasting.horizon import TimeHorizon
from eie.forecasting.load import LoadForecast
from eie.forecasting.solar import PVForecast
from eie.mpc.schedule import MPCSchedule, ScheduledAction
from eie.mpc.scheduler import MPCScheduler
from eie.optimization.optimizer import ShadowOptimizer
from eie.optimization.snapshot import (
    BatteryConstraints,
    GridConstraints,
    SiteSnapshot,
    ThermalStorageConstraints,
)
from eie.reference.state import ReferenceState


@pytest.fixture
def noon_utc() -> datetime:
    return datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def reference(noon_utc: datetime) -> ReferenceState:
    return ReferenceState(
        reference_state_id="ref-mpc-test",
        timestamp=noon_utc,
        ambient_temperature_k=293.15,
        ambient_pressure_pa=101_325.0,
        confidence=0.99,
        valid_until=noon_utc + timedelta(hours=12),
    )


@pytest.fixture
def boundary(reference: ReferenceState, noon_utc: datetime) -> Boundary:
    return Boundary(
        boundary_id="bnd-mpc-test",
        boundary_type=BoundaryType.SITE,
        included_entity_ids=["battery", "ts"],
        excluded_entity_ids=[],
        reference_state_id=reference.reference_state_id,
        accounting_period_start=noon_utc,
        accounting_period_end=noon_utc + timedelta(hours=12),
    )


@pytest.fixture
def meta(reference: ReferenceState, boundary: Boundary, noon_utc: datetime) -> Metadata:
    return Metadata(
        timestamp=noon_utc, source="mpc-test", method=MeasurementMethod.SIMULATED,
        unit="J", confidence=1.0, uncertainty=0.0,
        boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id,
    )


@pytest.fixture
def snapshot(reference: ReferenceState, boundary: Boundary, meta: Metadata,
             noon_utc: datetime) -> SiteSnapshot:
    battery_state = BatteryState(
        storage_id="b1", stored_energy_j=3_600_000.0 * 5, soc=0.60, soh=0.95,
        reserve_energy_j=3_600_000.0 * 0.5,
        boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id,
        metadata=meta,
    )
    ts_state = ThermalStorageState(
        storage_id="ts1",
        layers=[ThermalLayer(temperature_k=333.15, energy_j=2_000_000.0)],
        boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id,
        metadata=meta,
    )
    return SiteSnapshot(
        snapshot_id="snap-mpc-001", timestamp=noon_utc,
        reference_state=reference, boundary=boundary,
        horizon_duration_s=3600.0, available_pv_power_w=5_000.0,
        battery_state=battery_state,
        battery_constraints=BatteryConstraints(
            max_charge_power_w=3_000.0, max_discharge_power_w=3_000.0,
            min_soc=0.10, max_soc=0.95, round_trip_efficiency=0.92,
            capacity_j=3_600_000.0 * 10.0,
        ),
        thermal_storage_state=ts_state,
        thermal_storage_constraints=ThermalStorageConstraints(
            max_charge_power_w=2_000.0, max_discharge_power_w=2_000.0,
            charge_efficiency=0.95, discharge_efficiency=0.95, standby_loss_w=50.0,
        ),
        building_electric_load_w=1_200.0, heat_pump_rated_cop=3.5,
        heat_pump_max_power_w=2_000.0, delivered_heat_temperature_k=318.15,
        heat_demand_w=3_000.0,
        grid_constraints=GridConstraints(
            import_limit_w=10_000.0, export_limit_w=5_000.0,
            import_tariff_per_kwh=0.28, export_tariff_per_kwh=0.05,
            carbon_intensity_kg_per_kwh=0.42,
        ),
    )


@pytest.fixture
def horizon_4h(noon_utc: datetime) -> TimeHorizon:
    return TimeHorizon(start=noon_utc, step_duration_s=3600.0, n_steps=4)


@pytest.fixture
def bundle(horizon_4h: TimeHorizon) -> "ForecastBundle":  # type: ignore[name-defined]
    from eie.forecasting.bundle import ForecastBundle
    pv = PVForecast(panel_area_m2=20.0, panel_efficiency=0.20)
    lf = LoadForecast(base_thermal_w=5000.0, base_electrical_w=2000.0)
    return make_bundle(horizon_4h, pv, lf, reference_temperature_k=293.15)


@pytest.fixture
def scheduler() -> MPCScheduler:
    optimizer = ShadowOptimizer(grid_points_per_variable=5)
    policies = [
        BatteryPolicy(protected_reserve_fraction=0.20),
        ThermalStoragePolicy(min_comfort_k=293.15, max_comfort_k=340.0),
    ]
    gate = SafetyGate()
    return MPCScheduler(optimizer=optimizer, policies=policies, gate=gate)


def test_schedule_step_count(scheduler: MPCScheduler, snapshot: SiteSnapshot,
                              bundle: "ForecastBundle", noon_utc: datetime) -> None:  # type: ignore[name-defined]
    schedule = scheduler.schedule(snapshot, bundle, created_at=noon_utc)
    assert isinstance(schedule, MPCSchedule)
    assert len(schedule.steps) == bundle.horizon.n_steps


def test_schedule_snapshot_id(scheduler: MPCScheduler, snapshot: SiteSnapshot,
                               bundle: "ForecastBundle", noon_utc: datetime) -> None:  # type: ignore[name-defined]
    schedule = scheduler.schedule(snapshot, bundle, created_at=noon_utc)
    assert schedule.snapshot_id == snapshot.snapshot_id


def test_schedule_total_improvement_non_negative(scheduler: MPCScheduler,
                                                   snapshot: SiteSnapshot,
                                                   bundle: "ForecastBundle",  # type: ignore[name-defined]
                                                   noon_utc: datetime) -> None:
    schedule = scheduler.schedule(snapshot, bundle, created_at=noon_utc)
    assert schedule.total_forecast_exergy_improvement_j >= 0.0


def test_schedule_is_frozen(scheduler: MPCScheduler, snapshot: SiteSnapshot,
                             bundle: "ForecastBundle", noon_utc: datetime) -> None:  # type: ignore[name-defined]
    schedule = scheduler.schedule(snapshot, bundle, created_at=noon_utc)
    with pytest.raises((AttributeError, TypeError)):
        schedule.snapshot_id = "tampered"  # type: ignore[misc]


def test_scheduled_action_per_step(scheduler: MPCScheduler, snapshot: SiteSnapshot,
                                    bundle: "ForecastBundle", noon_utc: datetime) -> None:  # type: ignore[name-defined]
    schedule = scheduler.schedule(snapshot, bundle, created_at=noon_utc)
    for step_action in schedule.steps:
        assert isinstance(step_action, ScheduledAction)
        assert len(step_action.gate_verdicts) == len(step_action.control_actions)


def test_verdicts_match_actions(scheduler: MPCScheduler, snapshot: SiteSnapshot,
                                 bundle: "ForecastBundle", noon_utc: datetime) -> None:  # type: ignore[name-defined]
    schedule = scheduler.schedule(snapshot, bundle, created_at=noon_utc)
    for sa in schedule.steps:
        action_ids = {a.action_id for a in sa.control_actions}
        verdict_ids = {v.action_id for v in sa.gate_verdicts}
        assert action_ids == verdict_ids


def test_approved_actions_have_approved_verdicts(scheduler: MPCScheduler,
                                                   snapshot: SiteSnapshot,
                                                   bundle: "ForecastBundle",  # type: ignore[name-defined]
                                                   noon_utc: datetime) -> None:
    schedule = scheduler.schedule(snapshot, bundle, created_at=noon_utc)
    for sa in schedule.steps:
        approved_ids = {a.action_id for a in sa.approved_actions}
        for v in sa.gate_verdicts:
            if v.action_id in approved_ids:
                assert v.approved


def test_first_step_actions_only_approved(scheduler: MPCScheduler,
                                           snapshot: SiteSnapshot,
                                           bundle: "ForecastBundle",  # type: ignore[name-defined]
                                           noon_utc: datetime) -> None:
    schedule = scheduler.schedule(snapshot, bundle, created_at=noon_utc)
    first_step = schedule.steps[0]
    for a in schedule.first_step_actions:
        assert a.action_id in {v.action_id for v in first_step.gate_verdicts if v.approved}


def test_approved_steps_property(scheduler: MPCScheduler, snapshot: SiteSnapshot,
                                  bundle: "ForecastBundle", noon_utc: datetime) -> None:  # type: ignore[name-defined]
    schedule = scheduler.schedule(snapshot, bundle, created_at=noon_utc)
    approved = schedule.approved_steps
    assert all(s.approved for s in approved)


def test_schedule_length_mismatch(horizon_4h: TimeHorizon, noon_utc: datetime) -> None:
    from eie.forecasting.horizon import ForecastStep
    from eie.optimization.optimizer import OptimizationResult, SIMPLE_SITE_DISPATCH_SPACE
    from eie.optimization.objective import STANDARD_OBJECTIVES
    dummy_step = ForecastStep(
        step_index=0,
        start=noon_utc,
        end=noon_utc + timedelta(hours=1),
        duration_s=3600.0,
    )
    opt = OptimizationResult(
        result_id="r", timestamp=noon_utc, snapshot_id="s",
        feasible_decisions=[], pareto_front=[], recommended_decision=None,
        objective_scores={}, evaluated_count=0, infeasible_count=0,
        guard_bypass_count=0, objectives=list(STANDARD_OBJECTIVES),
    )
    sa = ScheduledAction(
        step=dummy_step, optimization_result=opt,
        control_actions=[], gate_verdicts=[], approved=False,
        forecast_exergy_improvement_j=0.0,
    )
    with pytest.raises(ValueError, match="n_steps"):
        MPCSchedule(
            schedule_id="s1", created_at=noon_utc,
            horizon=horizon_4h, snapshot_id="snap",
            steps=[sa],
            total_forecast_exergy_improvement_j=0.0,
        )
