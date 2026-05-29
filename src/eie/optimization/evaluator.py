"""Guard-verified objective evaluator for advisory dispatch decisions.

The evaluator is the only component that touches both the physics kernel and
the guard stack.  Optimization logic must never bypass guard checks;
any candidate that fails a guard is unconditionally rejected.

For every candidate dispatch point, the evaluator:
1. Computes the exergy balance by simulating one time step.
2. Runs the full guard stack (Reference, Boundary, Physics, FalseExergyGain).
3. If any guard fails → marks the candidate infeasible with the guard reason.
4. If all guards pass → evaluates all objectives and returns the score.

No candidate that fails any guard is allowed into the feasible set.

The evaluator has NO side effects — it does not write to the ledger, does not
push recommendations anywhere, and does not actuate hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from eie.core.enums import BoundaryType, Carrier, MeasurementMethod
from eie.core.errors import DomainError
from eie.exergy.cooling import cooling_service_exergy_rate
from eie.exergy.electrical import electrical_exergy_rate
from eie.exergy.heat import heat_exergy_rate
from eie.exergy.kernel import ExergyKernelV0, energy_balance_residual, exergy_balance_residual
from eie.exergy.storage import battery_stored_exergy
from eie.guards.boundary_guard import BoundaryGuard
from eie.guards.false_exergy_gain import FalseExergyGainGuard
from eie.guards.physics_guard import GuardResult, PhysicsGuard
from eie.guards.reference_guard import ReferenceGuard
from eie.ledger.entries import LedgerEntry
from eie.optimization.decision import DispatchDecision, FeasibilityNote
from eie.optimization.objective import (
    MultiObjectiveScore,
    ObjectiveValue,
    OptimizationObjective,
)
from eie.optimization.snapshot import SiteSnapshot


JOULES_PER_KWH = 3_600_000.0


@dataclass(frozen=True)
class SimulatedOutcome:
    """Physics outcome of simulating one dispatch decision for one horizon."""

    # Energy flows [J]
    pv_energy_j: float
    battery_charge_j: float
    battery_discharge_j: float
    heat_pump_energy_in_j: float
    heat_pump_heat_out_j: float
    grid_import_j: float
    grid_export_j: float
    building_load_j: float
    thermal_served_from_hp_j: float
    thermal_served_from_storage_j: float
    thermal_unmet_j: float

    # Exergy flows [J]
    pv_exergy_j: float
    heat_pump_useful_exergy_j: float
    building_useful_exergy_j: float
    battery_exergy_stored_delta_j: float
    thermal_storage_exergy_delta_j: float
    destroyed_exergy_j: float

    # State at end of horizon
    battery_soc_end: float

    # Cost and carbon
    grid_import_cost_currency: float
    grid_carbon_kg: float

    # Guard results
    guard_results: list[GuardResult]

    # Ledger entry for this simulated period
    ledger_entry: LedgerEntry


def _make_ledger_entry(
    *,
    decision_id: str,
    timestamp: datetime,
    boundary_id: str,
    reference_state_id: str,
    reference_temperature_k: float,
    outcome: SimulatedOutcome,
) -> LedgerEntry:
    # Energy balance (electrical bus):
    #   in  = PV + battery_discharge + grid_import
    #   out = building + grid_export + HP_electrical_in
    #   stored = battery_charge (energy drawn from bus to battery charger input)
    energy_in = outcome.pv_energy_j + outcome.battery_discharge_j + outcome.grid_import_j
    energy_out = outcome.building_load_j + outcome.grid_export_j + outcome.heat_pump_energy_in_j
    energy_stored = outcome.battery_charge_j
    energy_rejected = 0.0
    energy_res = energy_balance_residual(energy_in, energy_out, energy_stored, energy_rejected)

    # Exergy balance:
    #   in       = PV exergy + grid_import exergy (electricity quality=1)
    #   useful   = building + HP heat + grid_export (electricity exported at quality=1)
    #   stored   = battery exergy stored
    #   destroyed = HP irreversibilities + battery charging losses
    grid_import_exergy_j = outcome.grid_import_j    # electricity: quality factor = 1.0
    grid_export_exergy_j = outcome.grid_export_j    # exported electricity is useful output
    exergy_in = outcome.pv_exergy_j + grid_import_exergy_j
    useful_exergy = (outcome.heat_pump_useful_exergy_j
                     + outcome.building_useful_exergy_j
                     + grid_export_exergy_j)
    stored_exergy_delta = outcome.battery_exergy_stored_delta_j
    recovered = 0.0
    rejected = 0.0
    destroyed = max(0.0, outcome.destroyed_exergy_j)
    exergy_res = exergy_balance_residual(
        exergy_in, useful_exergy, stored_exergy_delta, recovered, rejected, destroyed
    )
    entropy = destroyed / reference_temperature_k if reference_temperature_k > 0 else 0.0
    return LedgerEntry(
        ledger_id=f"opt-sim:{decision_id}",
        timestamp=timestamp,
        boundary_id=boundary_id,
        reference_state_id=reference_state_id,
        energy_in_j=energy_in,
        energy_out_j=energy_out,
        energy_stored_delta_j=energy_stored,
        energy_rejected_j=energy_rejected,
        energy_residual_j=energy_res,
        exergy_in_j=exergy_in,
        useful_exergy_j=useful_exergy,
        stored_exergy_delta_j=stored_exergy_delta,
        recovered_exergy_j=recovered,
        rejected_exergy_j=rejected,
        destroyed_exergy_j=destroyed,
        exergy_residual_j=exergy_res,
        entropy_generated_j_per_k=entropy,
        flags=["advisory_simulation"],
        confidence=0.90,
    )


def simulate_dispatch(
    snapshot: SiteSnapshot,
    variable_values: dict[str, float],
    *,
    timestamp: datetime,
    decision_id: str,
) -> SimulatedOutcome:
    """Simulate one dispatch decision and return the physical outcome.

    Variable names used:
    - "heat_pump_fraction"       : fraction of HP rated power actually used [0, 1]
    - "battery_charge_fraction"  : fraction of available PV sent to battery [0, 1]
    - "thermal_storage_fraction" : fraction of thermal demand served from storage [0, 1]

    Physical constraints enforced:
    - PV power split: heat_pump + battery_charge ≤ available PV
    - Thermal demand balance: hp_out + ts_discharge ≥ demand (excess = 0)
    - Battery SOC remains within constraints
    """
    from eie.core.tolerance import DEFAULT_TOLERANCE

    T0 = snapshot.reference_temperature_k
    dt = snapshot.horizon_duration_s

    # ── Unpack dispatch variables ───────────────────────────────────────────
    hp_fraction = max(0.0, min(1.0, variable_values.get("heat_pump_fraction", 0.0)))
    batt_fraction = max(0.0, min(1.0, variable_values.get("battery_charge_fraction", 0.0)))
    ts_fraction = max(0.0, min(1.0, variable_values.get("thermal_storage_fraction", 0.0)))

    # ── PV dispatch ─────────────────────────────────────────────────────────
    pv_w = snapshot.available_pv_power_w
    hp_power_w = min(hp_fraction * snapshot.heat_pump_max_power_w, pv_w)
    pv_remaining_after_hp = pv_w - hp_power_w

    batt_charge_w = min(
        batt_fraction * snapshot.battery_constraints.max_charge_power_w,
        pv_remaining_after_hp,
    )
    # After HP and battery, remaining PV covers building load then exports surplus
    pv_remaining_after_batt = pv_remaining_after_hp - batt_charge_w
    building_load_w = snapshot.building_electric_load_w
    grid_import_w = max(0.0, building_load_w - pv_remaining_after_batt)
    grid_export_w = max(0.0, pv_remaining_after_batt - building_load_w)

    # ── Heat pump output ─────────────────────────────────────────────────────
    hp_heat_w = hp_power_w * snapshot.heat_pump_rated_cop

    # ── Thermal storage ──────────────────────────────────────────────────────
    thermal_demand_w = snapshot.heat_demand_w
    ts_discharge_w = min(
        ts_fraction * snapshot.thermal_storage_constraints.max_discharge_power_w,
        thermal_demand_w,
        snapshot.thermal_storage_constraints.max_discharge_power_w,
    )
    thermal_from_hp = min(hp_heat_w, thermal_demand_w - ts_discharge_w)
    thermal_unmet_w = max(0.0, thermal_demand_w - ts_discharge_w - thermal_from_hp)

    # ── Battery state update ─────────────────────────────────────────────────
    bc = snapshot.battery_constraints
    bs = snapshot.battery_state
    # Electrical energy drawn from bus to battery input
    batt_charge_electrical_j = batt_charge_w * dt
    # Energy actually stored after charging efficiency
    charge_stored_j = batt_charge_electrical_j * bc.round_trip_efficiency
    usable_charge = bc.usable_charge_j(bs.soc)
    actual_charge_stored_j = min(charge_stored_j, usable_charge)
    battery_soc_end = bs.soc + actual_charge_stored_j / bc.capacity_j if bc.capacity_j > 0 else bs.soc
    battery_soc_end = max(bc.min_soc, min(bc.max_soc, battery_soc_end))

    # ── Energy tallies [J] ──────────────────────────────────────────────────
    pv_energy_j = pv_w * dt
    batt_charge_j = batt_charge_w * dt          # electrical energy from bus to battery
    batt_discharge_j = 0.0                       # v0 does not discharge battery in same period
    hp_energy_in_j = hp_power_w * dt
    hp_heat_out_j = hp_heat_w * dt
    grid_import_j = grid_import_w * dt
    grid_export_j = grid_export_w * dt
    building_load_j = building_load_w * dt
    thermal_from_hp_j = thermal_from_hp * dt
    thermal_from_storage_j = ts_discharge_w * dt
    thermal_unmet_j = thermal_unmet_w * dt

    # ── Exergy tallies [J] ──────────────────────────────────────────────────
    pv_exergy_j = electrical_exergy_rate(pv_w) * dt
    hp_useful_exergy_j = heat_exergy_rate(
        hp_heat_w, snapshot.delivered_heat_temperature_k, T0
    ) * dt
    building_useful_exergy_j = electrical_exergy_rate(building_load_w) * dt
    # Exergy stored in battery = energy stored (electrical storage: quality=1)
    battery_exergy_stored_j = actual_charge_stored_j

    # Exergy destruction accounting:
    # HP: electrical in − useful thermal exergy out (COP multiplies energy but not exergy)
    hp_destroyed = max(0.0, hp_energy_in_j - hp_useful_exergy_j)
    # Battery charging losses: electrical drawn from bus − energy actually stored
    batt_destroyed = max(0.0, batt_charge_j - battery_exergy_stored_j)
    destroyed_exergy_j = hp_destroyed + batt_destroyed

    # ── Cost and carbon ──────────────────────────────────────────────────────
    gc = snapshot.grid_constraints
    grid_import_kwh = grid_import_j / JOULES_PER_KWH
    cost = grid_import_kwh * gc.import_tariff_per_kwh
    carbon = grid_import_kwh * gc.carbon_intensity_kg_per_kwh

    # ── Build ledger entry ───────────────────────────────────────────────────
    ledger_entry = _make_ledger_entry(
        decision_id=decision_id,
        timestamp=timestamp,
        boundary_id=snapshot.boundary.boundary_id,
        reference_state_id=snapshot.reference_state.reference_state_id,
        reference_temperature_k=T0,
        outcome=SimulatedOutcome(
            pv_energy_j=pv_energy_j,
            battery_charge_j=batt_charge_j,
            battery_discharge_j=batt_discharge_j,
            heat_pump_energy_in_j=hp_energy_in_j,
            heat_pump_heat_out_j=hp_heat_out_j,
            grid_import_j=grid_import_j,
            grid_export_j=grid_export_j,
            building_load_j=building_load_j,
            thermal_served_from_hp_j=thermal_from_hp_j,
            thermal_served_from_storage_j=thermal_from_storage_j,
            thermal_unmet_j=thermal_unmet_j,
            pv_exergy_j=pv_exergy_j,
            heat_pump_useful_exergy_j=hp_useful_exergy_j,
            building_useful_exergy_j=building_useful_exergy_j,
            battery_exergy_stored_delta_j=battery_exergy_stored_j,
            thermal_storage_exergy_delta_j=0.0,
            destroyed_exergy_j=destroyed_exergy_j,
            battery_soc_end=battery_soc_end,
            grid_import_cost_currency=cost,
            grid_carbon_kg=carbon,
            guard_results=[],
            ledger_entry=LedgerEntry(
                ledger_id="placeholder",
                timestamp=timestamp,
                boundary_id=snapshot.boundary.boundary_id,
                reference_state_id=snapshot.reference_state.reference_state_id,
                energy_in_j=0.0,
                energy_out_j=0.0,
                energy_stored_delta_j=0.0,
                energy_rejected_j=0.0,
                energy_residual_j=0.0,
                exergy_in_j=0.0,
                useful_exergy_j=0.0,
                stored_exergy_delta_j=0.0,
                recovered_exergy_j=0.0,
                rejected_exergy_j=0.0,
                destroyed_exergy_j=0.0,
                exergy_residual_j=0.0,
                entropy_generated_j_per_k=0.0,
            ),
        ),
    )

    # Run guards
    guard_results: list[GuardResult] = [
        ReferenceGuard().check(snapshot.reference_state, at=timestamp),
        BoundaryGuard().check(
            snapshot.boundary,
            expected_reference_state_id=snapshot.reference_state.reference_state_id,
        ),
        PhysicsGuard().check_ledger_entry(ledger_entry),
        FalseExergyGainGuard().check(
            exergy_in_j=pv_exergy_j,
            useful_exergy_j=hp_useful_exergy_j + building_useful_exergy_j,
            stored_exergy_delta_j=battery_exergy_stored_j,
            recovered_exergy_j=0.0,
            destroyed_exergy_j=destroyed_exergy_j,
            entropy_generated_j_per_k=destroyed_exergy_j / T0 if T0 > 0 else 0.0,
            reference_state=snapshot.reference_state,
            boundary=snapshot.boundary,
            at=timestamp,
            carrier=Carrier.ELECTRIC,
            quality_factor=1.0,
        ),
    ]

    return SimulatedOutcome(
        pv_energy_j=pv_energy_j,
        battery_charge_j=batt_charge_j,
        battery_discharge_j=batt_discharge_j,
        heat_pump_energy_in_j=hp_energy_in_j,
        heat_pump_heat_out_j=hp_heat_out_j,
        grid_import_j=grid_import_j,
        grid_export_j=grid_export_j,
        building_load_j=building_load_j,
        thermal_served_from_hp_j=thermal_from_hp_j,
        thermal_served_from_storage_j=thermal_from_storage_j,
        thermal_unmet_j=thermal_unmet_j,
        pv_exergy_j=pv_exergy_j,
        heat_pump_useful_exergy_j=hp_useful_exergy_j,
        building_useful_exergy_j=building_useful_exergy_j,
        battery_exergy_stored_delta_j=battery_exergy_stored_j,
        thermal_storage_exergy_delta_j=0.0,
        destroyed_exergy_j=destroyed_exergy_j,
        battery_soc_end=battery_soc_end,
        grid_import_cost_currency=cost,
        grid_carbon_kg=carbon,
        guard_results=guard_results,
        ledger_entry=ledger_entry,
    )


def evaluate_objectives(
    outcome: SimulatedOutcome,
    objectives: Sequence[OptimizationObjective],
) -> MultiObjectiveScore | None:
    """Return objective scores for a simulated outcome.

    Returns None if any guard failed (infeasible candidate).
    """
    if any(not r.passed for r in outcome.guard_results):
        return None

    exergy_in = outcome.pv_exergy_j
    useful_exergy = outcome.heat_pump_useful_exergy_j + outcome.building_useful_exergy_j
    exergy_efficiency = useful_exergy / exergy_in if exergy_in > 1.0e-9 else 0.0

    obj_map = {
        "exergy_destruction_j":   outcome.destroyed_exergy_j,
        "exergy_efficiency":       exergy_efficiency,
        "marginal_carbon_kg":      outcome.grid_carbon_kg,
        "operating_cost_currency": outcome.grid_import_cost_currency,
        "battery_soc_end":         outcome.battery_soc_end,
    }

    values: dict[str, ObjectiveValue] = {}
    for obj in objectives:
        if obj.name in obj_map:
            values[obj.name] = ObjectiveValue(
                objective_name=obj.name,
                value=obj_map[obj.name],
                unit=obj.unit,
            )
        else:
            raise DomainError(
                f"objective {obj.name!r} is not computable from SimulatedOutcome; "
                f"available: {sorted(obj_map)}"
            )
    return MultiObjectiveScore(decision_id="", values=values)
