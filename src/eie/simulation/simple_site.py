"""Deterministic simple-site simulation for Exergy Kernel v0."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from eie.boundary.boundary import Boundary
from eie.core.constants import SECONDS_PER_HOUR, STANDARD_ATMOSPHERE_PA
from eie.core.enums import BoundaryType, Carrier, MeasurementMethod
from eie.exergy.electrical import electrical_exergy_rate
from eie.exergy.heat import finite_stream_heat_exergy_rate, heat_exergy_rate
from eie.exergy.kernel import ExergyKernelV0, energy_balance_residual, exergy_balance_residual
from eie.exergy.storage import thermal_storage_exergy
from eie.flows.electrical import ElectricalFlow
from eie.flows.storage import BatteryState, ThermalLayer, ThermalStorageState
from eie.guards.boundary_guard import BoundaryGuard
from eie.guards.false_exergy_gain import FalseExergyGainGuard
from eie.guards.physics_guard import GuardResult, PhysicsGuard
from eie.guards.reference_guard import ReferenceGuard
from eie.ledger.entries import AuditLedger, LedgerEntry
from eie.reference.state import ReferenceState


@dataclass(frozen=True)
class SimpleSiteReport:
    reference_state: ReferenceState
    boundary: Boundary
    pv_exergy_j: float
    battery_stored_exergy_j: float
    heat_pump_useful_thermal_exergy_j: float
    resistance_heat_exergy_for_same_input_j: float
    waste_heat_recoverable_exergy_j: float
    thermal_storage_exergy_j: float
    ledger: AuditLedger
    guard_results: list[GuardResult]
    impossible_case_guard_results: list[GuardResult]
    recommendations: list[str]

    def to_text(self) -> str:
        normal_status = "PASS" if all(result.passed for result in self.guard_results) else "FAIL"
        impossible_status = (
            "PASS"
            if any(not result.passed for result in self.impossible_case_guard_results)
            else "FAIL"
        )
        lines = [
            "Exergy Kernel v0 simple-site report",
            f"reference_state_id: {self.reference_state.reference_state_id}",
            f"boundary_id: {self.boundary.boundary_id}",
            f"pv_exergy_j: {self.pv_exergy_j:.3f}",
            f"battery_stored_exergy_j: {self.battery_stored_exergy_j:.3f}",
            f"heat_pump_useful_thermal_exergy_j: {self.heat_pump_useful_thermal_exergy_j:.3f}",
            f"resistance_heat_exergy_for_same_input_j: {self.resistance_heat_exergy_for_same_input_j:.3f}",
            f"waste_heat_recoverable_exergy_j: {self.waste_heat_recoverable_exergy_j:.3f}",
            f"thermal_storage_exergy_j: {self.thermal_storage_exergy_j:.3f}",
            f"ledger_entries: {len(self.ledger.entries)}",
            f"normal_guard_status: {normal_status}",
            f"impossible_case_detected: {impossible_status}",
            "recommendations:",
        ]
        lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


def _entry(
    *,
    ledger_id: str,
    timestamp: datetime,
    boundary: Boundary,
    energy_in_j: float,
    energy_out_j: float,
    energy_stored_delta_j: float,
    energy_rejected_j: float,
    exergy_in_j: float,
    useful_exergy_j: float,
    stored_exergy_delta_j: float,
    recovered_exergy_j: float,
    rejected_exergy_j: float,
    destroyed_exergy_j: float,
    reference_temperature_k: float,
    flags: list[str] | None = None,
) -> LedgerEntry:
    entropy_generated = destroyed_exergy_j / reference_temperature_k
    return LedgerEntry(
        ledger_id=ledger_id,
        timestamp=timestamp,
        boundary_id=boundary.boundary_id,
        reference_state_id=boundary.reference_state_id,
        energy_in_j=energy_in_j,
        energy_out_j=energy_out_j,
        energy_stored_delta_j=energy_stored_delta_j,
        energy_rejected_j=energy_rejected_j,
        energy_residual_j=energy_balance_residual(
            energy_in_j,
            energy_out_j,
            energy_stored_delta_j,
            energy_rejected_j,
        ),
        exergy_in_j=exergy_in_j,
        useful_exergy_j=useful_exergy_j,
        stored_exergy_delta_j=stored_exergy_delta_j,
        recovered_exergy_j=recovered_exergy_j,
        rejected_exergy_j=rejected_exergy_j,
        destroyed_exergy_j=destroyed_exergy_j,
        exergy_residual_j=exergy_balance_residual(
            exergy_in_j,
            useful_exergy_j,
            stored_exergy_delta_j,
            recovered_exergy_j,
            rejected_exergy_j,
            destroyed_exergy_j,
        ),
        entropy_generated_j_per_k=entropy_generated,
        flags=[] if flags is None else flags,
        confidence=0.95,
    )


def run_simple_site(at: datetime | None = None) -> SimpleSiteReport:
    """Run a deterministic one-hour advisory simulation."""

    at = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc) if at is None else at
    reference = ReferenceState(
        reference_state_id="ref-ambient-298k",
        timestamp=at,
        ambient_temperature_k=298.15,
        ambient_pressure_pa=STANDARD_ATMOSPHERE_PA,
        relative_humidity=0.45,
        nominal_grid_voltage_v=230.0,
        nominal_grid_frequency_hz=50.0,
        marginal_carbon_kg_per_kwh=0.42,
        marginal_price_per_kwh=0.18,
        confidence=0.98,
        valid_until=at + timedelta(minutes=30),
        notes="simple deterministic v0 simulation reference",
    )
    boundary = Boundary(
        boundary_id="site-boundary-v0",
        boundary_type=BoundaryType.SITE,
        included_entity_ids=["pv", "battery", "heat-pump", "thermal-store", "building", "waste-stream"],
        excluded_entity_ids=[],
        reference_state_id=reference.reference_state_id,
        accounting_period_start=at,
        accounting_period_end=at + timedelta(hours=1),
    )
    kernel = ExergyKernelV0(reference_state=reference, boundary=boundary)
    power_metadata = kernel.metadata(
        timestamp=at,
        source="simple_site",
        method=MeasurementMethod.SIMULATED,
        unit="W",
        confidence=0.95,
        uncertainty=0.0,
    )
    energy_metadata = kernel.metadata(
        timestamp=at,
        source="simple_site",
        method=MeasurementMethod.SIMULATED,
        unit="J",
        confidence=0.95,
        uncertainty=0.0,
    )

    duration_s = SECONDS_PER_HOUR
    pv_power_w = 5_000.0
    hp_electric_power_w = 1_500.0
    site_electric_service_w = 1_500.0
    battery_charge_power_w = 2_000.0
    hp_cop_heating = 3.0
    hp_heat_output_w = hp_electric_power_w * hp_cop_heating
    delivered_heat_temperature_k = 318.15
    low_grade_resistance_temperature_k = 318.15
    waste_mass_flow_kg_s = 0.08
    waste_cp_j_kg_k = 4_180.0
    waste_t_in_k = 350.0
    waste_t_out_k = 320.0

    pv_flow = ElectricalFlow(
        flow_id="pv-real-power",
        real_power_w=pv_power_w,
        voltage_v=230.0,
        frequency_hz=50.0,
        power_factor=1.0,
        thd=0.02,
        availability_factor=0.99,
        boundary_id=boundary.boundary_id,
        reference_state_id=reference.reference_state_id,
        metadata=power_metadata,
    )
    kernel.electrical_flow(pv_flow)

    pv_exergy_j = electrical_exergy_rate(pv_power_w) * duration_s
    battery_state = BatteryState(
        storage_id="battery-v0",
        stored_energy_j=battery_charge_power_w * duration_s,
        soc=0.60,
        soh=0.96,
        reserve_energy_j=1_000.0 * duration_s,
        boundary_id=boundary.boundary_id,
        reference_state_id=reference.reference_state_id,
        metadata=energy_metadata,
    )
    battery_stored_exergy_j = kernel.battery_state_exergy(battery_state)
    heat_pump_useful_thermal_exergy_j = (
        heat_exergy_rate(hp_heat_output_w, delivered_heat_temperature_k, reference.ambient_temperature_k)
        * duration_s
    )
    resistance_heat_exergy_for_same_input_j = (
        heat_exergy_rate(hp_electric_power_w, low_grade_resistance_temperature_k, reference.ambient_temperature_k)
        * duration_s
    )
    waste_heat_recoverable_exergy_j = (
        finite_stream_heat_exergy_rate(
            waste_mass_flow_kg_s,
            waste_cp_j_kg_k,
            waste_t_in_k,
            waste_t_out_k,
            reference.ambient_temperature_k,
        )
        * duration_s
    )
    thermal_storage_state = ThermalStorageState(
        storage_id="stratified-water-store-v0",
        layers=[
            ThermalLayer(temperature_k=333.15, energy_j=3_000_000.0),
            ThermalLayer(temperature_k=318.15, energy_j=3_500_000.0),
            ThermalLayer(temperature_k=303.15, energy_j=2_500_000.0),
        ],
        boundary_id=boundary.boundary_id,
        reference_state_id=reference.reference_state_id,
        metadata=energy_metadata,
    )
    thermal_storage_exergy_j = thermal_storage_exergy(
        thermal_storage_state.layers,
        reference.ambient_temperature_k,
    )

    ledger = AuditLedger()
    pv_entry = _entry(
        ledger_id="ledger-pv-allocation-hour-1",
        timestamp=at,
        boundary=boundary,
        energy_in_j=pv_power_w * duration_s,
        energy_out_j=(hp_electric_power_w + site_electric_service_w) * duration_s,
        energy_stored_delta_j=battery_charge_power_w * duration_s,
        energy_rejected_j=0.0,
        exergy_in_j=pv_exergy_j,
        useful_exergy_j=(hp_electric_power_w + site_electric_service_w) * duration_s,
        stored_exergy_delta_j=battery_charge_power_w * duration_s,
        recovered_exergy_j=0.0,
        rejected_exergy_j=0.0,
        destroyed_exergy_j=0.0,
        reference_temperature_k=reference.ambient_temperature_k,
        flags=["electrical_exergy_high_grade", "service_derating_tracked_separately"],
    )
    ledger.append(pv_entry)

    heat_pump_energy_in_j = hp_electric_power_w * duration_s + (hp_heat_output_w - hp_electric_power_w) * duration_s
    heat_pump_destroyed_exergy_j = hp_electric_power_w * duration_s - heat_pump_useful_thermal_exergy_j
    hp_entry = _entry(
        ledger_id="ledger-heat-pump-hour-1",
        timestamp=at,
        boundary=boundary,
        energy_in_j=heat_pump_energy_in_j,
        energy_out_j=hp_heat_output_w * duration_s,
        energy_stored_delta_j=0.0,
        energy_rejected_j=0.0,
        exergy_in_j=hp_electric_power_w * duration_s,
        useful_exergy_j=heat_pump_useful_thermal_exergy_j,
        stored_exergy_delta_j=0.0,
        recovered_exergy_j=0.0,
        rejected_exergy_j=0.0,
        destroyed_exergy_j=heat_pump_destroyed_exergy_j,
        reference_temperature_k=reference.ambient_temperature_k,
        flags=["ambient_heat_energy_included", "ambient_heat_exergy_not_counted_as_free_gain"],
    )
    ledger.append(hp_entry)

    waste_entry = _entry(
        ledger_id="ledger-waste-heat-hour-1",
        timestamp=at,
        boundary=boundary,
        energy_in_j=waste_mass_flow_kg_s * waste_cp_j_kg_k * (waste_t_in_k - waste_t_out_k) * duration_s,
        energy_out_j=0.0,
        energy_stored_delta_j=0.0,
        energy_rejected_j=waste_mass_flow_kg_s * waste_cp_j_kg_k * (waste_t_in_k - waste_t_out_k) * duration_s,
        exergy_in_j=waste_heat_recoverable_exergy_j,
        useful_exergy_j=0.0,
        stored_exergy_delta_j=0.0,
        recovered_exergy_j=0.0,
        rejected_exergy_j=waste_heat_recoverable_exergy_j,
        destroyed_exergy_j=0.0,
        reference_temperature_k=reference.ambient_temperature_k,
        flags=["recoverable_waste_heat_identified"],
    )
    ledger.append(waste_entry)

    guard_results: list[GuardResult] = [
        ReferenceGuard().check(reference, at=at),
        BoundaryGuard().check(boundary, expected_reference_state_id=reference.reference_state_id),
    ]
    guard_results.extend(PhysicsGuard().check_ledger(ledger.entries))
    guard_results.append(
        FalseExergyGainGuard().check(
            exergy_in_j=hp_entry.exergy_in_j,
            useful_exergy_j=hp_entry.useful_exergy_j,
            stored_exergy_delta_j=hp_entry.stored_exergy_delta_j,
            recovered_exergy_j=hp_entry.recovered_exergy_j,
            destroyed_exergy_j=hp_entry.destroyed_exergy_j,
            entropy_generated_j_per_k=hp_entry.entropy_generated_j_per_k,
            reference_state=reference,
            boundary=boundary,
            at=at,
            carrier=Carrier.THERMAL,
            quality_factor=hp_entry.useful_exergy_j / hp_entry.energy_out_j,
        )
    )

    impossible_case_guard_results = [
        FalseExergyGainGuard().check(
            exergy_in_j=1_000.0,
            useful_exergy_j=1_500.0,
            stored_exergy_delta_j=0.0,
            recovered_exergy_j=0.0,
            destroyed_exergy_j=-5.0,
            entropy_generated_j_per_k=-0.01,
            reference_state=reference,
            boundary=boundary,
            at=at,
            high_grade_exergy_out_j=2_000.0,
            low_grade_exergy_in_j=100.0,
            work_input_j=100.0,
        )
    ]

    recommendations = [
        "Battery electricity should not be wasted on low-grade heat when heat-pump or recovered-waste-heat service is available.",
        "Keep PV electricity available for high-grade work, reserves, or storage before down-converting it to low-grade heat.",
        "Treat recoverable waste heat as an audited opportunity, not as created energy.",
        "Do not authorize optimization if boundary, reference, residual, or false-gain guards fail.",
    ]
    return SimpleSiteReport(
        reference_state=reference,
        boundary=boundary,
        pv_exergy_j=pv_exergy_j,
        battery_stored_exergy_j=battery_stored_exergy_j,
        heat_pump_useful_thermal_exergy_j=heat_pump_useful_thermal_exergy_j,
        resistance_heat_exergy_for_same_input_j=resistance_heat_exergy_for_same_input_j,
        waste_heat_recoverable_exergy_j=waste_heat_recoverable_exergy_j,
        thermal_storage_exergy_j=thermal_storage_exergy_j,
        ledger=ledger,
        guard_results=guard_results,
        impossible_case_guard_results=impossible_case_guard_results,
        recommendations=recommendations,
    )


def main() -> None:
    print(run_simple_site().to_text())


if __name__ == "__main__":
    main()
