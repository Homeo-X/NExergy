"""Exergy Kernel v0 orchestration and balance residuals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from eie.boundary.boundary import Boundary
from eie.core.enums import Carrier, MeasurementMethod
from eie.core.errors import BoundaryError, MissingReferenceError
from eie.exergy.cooling import cooling_service_exergy_rate
from eie.exergy.electrical import electrical_exergy_rate, electrical_service_exergy_rate
from eie.exergy.heat import heat_exergy_rate
from eie.exergy.quality import classify_quality
from eie.exergy.storage import battery_stored_exergy, thermal_storage_exergy
from eie.flows.base import ExergyFlow, Metadata
from eie.flows.chemical import ChemicalFlow
from eie.flows.cooling import CoolingLoad
from eie.flows.electrical import ElectricalFlow
from eie.flows.storage import BatteryState, ThermalStorageState
from eie.flows.thermal import ThermalFlow
from eie.reference.state import ReferenceState


def energy_balance_residual(
    energy_in_j: float,
    energy_out_j: float,
    energy_stored_delta_j: float,
    energy_rejected_j: float,
    known_losses_j: float = 0.0,
) -> float:
    """Return first-law accounting residual."""

    return energy_in_j - energy_out_j - energy_stored_delta_j - energy_rejected_j - known_losses_j


def exergy_balance_residual(
    exergy_in_j: float,
    useful_exergy_j: float,
    stored_exergy_delta_j: float,
    recovered_exergy_j: float,
    rejected_exergy_j: float,
    destroyed_exergy_j: float,
) -> float:
    """Return exergy accounting residual."""

    return (
        exergy_in_j
        - useful_exergy_j
        - stored_exergy_delta_j
        - recovered_exergy_j
        - rejected_exergy_j
        - destroyed_exergy_j
    )


@dataclass(frozen=True)
class ExergyKernelV0:
    """Minimal trusted thermodynamic truth kernel.

    The kernel is deliberately advisory and computational only. It has no
    physical actuation surface.
    """

    reference_state: ReferenceState
    boundary: Boundary

    def __post_init__(self) -> None:
        if self.reference_state is None:
            raise MissingReferenceError("reference_state is required")
        if self.boundary is None:
            raise BoundaryError("boundary is required")
        if self.boundary.reference_state_id != self.reference_state.reference_state_id:
            raise BoundaryError(
                f"boundary reference {self.boundary.reference_state_id!r} does not match "
                f"kernel reference {self.reference_state.reference_state_id!r}"
            )

    @property
    def reference_temperature_k(self) -> float:
        return self.reference_state.ambient_temperature_k

    def _require_bound(self, boundary_id: str, reference_state_id: str) -> None:
        if boundary_id != self.boundary.boundary_id:
            raise BoundaryError(f"flow boundary {boundary_id!r} does not match kernel boundary")
        if reference_state_id != self.reference_state.reference_state_id:
            raise MissingReferenceError(
                f"flow reference {reference_state_id!r} does not match kernel reference"
            )

    def electrical_flow(self, flow: ElectricalFlow, *, service_derated: bool = False) -> ExergyFlow:
        self._require_bound(flow.boundary_id, flow.reference_state_id)
        exergy_rate = electrical_exergy_rate(flow.real_power_w)
        if service_derated:
            exergy_rate = electrical_service_exergy_rate(
                flow.real_power_w,
                voltage_quality=1.0,
                frequency_quality=1.0,
                harmonic_quality=1.0 if flow.thd is None else 1.0 - flow.thd,
                availability=1.0 if flow.availability_factor is None else flow.availability_factor,
            )
        quality_factor = 1.0 if flow.real_power_w == 0 else exergy_rate / flow.real_power_w
        return ExergyFlow(
            flow_id=f"{flow.flow_id}:exergy",
            carrier=Carrier.ELECTRIC,
            source_node_id="electrical_source",
            target_node_id="electrical_sink",
            energy_rate_w=flow.real_power_w,
            exergy_rate_w=exergy_rate,
            quality_factor=quality_factor,
            quality_grade=classify_quality(quality_factor, Carrier.ELECTRIC),
            boundary_id=flow.boundary_id,
            reference_state_id=flow.reference_state_id,
            metadata=flow.metadata,
            flags=[] if not service_derated else ["service_derated_not_thermodynamic_loss"],
        )

    def thermal_flow(self, flow: ThermalFlow) -> ExergyFlow:
        self._require_bound(flow.boundary_id, flow.reference_state_id)
        exergy_rate = heat_exergy_rate(flow.heat_rate_w, flow.source_temperature_k, self.reference_temperature_k)
        quality_factor = 0.0 if flow.heat_rate_w == 0 else exergy_rate / flow.heat_rate_w
        return ExergyFlow(
            flow_id=f"{flow.flow_id}:exergy",
            carrier=Carrier.THERMAL,
            source_node_id="thermal_source",
            target_node_id="thermal_sink",
            energy_rate_w=flow.heat_rate_w,
            exergy_rate_w=exergy_rate,
            quality_factor=quality_factor,
            quality_grade=classify_quality(quality_factor, Carrier.THERMAL),
            boundary_id=flow.boundary_id,
            reference_state_id=flow.reference_state_id,
            metadata=flow.metadata,
        )

    def cooling_load(self, load: CoolingLoad) -> ExergyFlow:
        self._require_bound(load.boundary_id, load.reference_state_id)
        exergy_rate = cooling_service_exergy_rate(
            load.cooling_rate_w,
            load.cold_temperature_k,
            self.reference_temperature_k,
        )
        quality_factor = 0.0 if load.cooling_rate_w == 0 else exergy_rate / load.cooling_rate_w
        return ExergyFlow(
            flow_id=f"{load.load_id}:exergy",
            carrier=Carrier.COOLING,
            source_node_id="cooled_space",
            target_node_id="cooling_service",
            energy_rate_w=load.cooling_rate_w,
            exergy_rate_w=exergy_rate,
            quality_factor=quality_factor,
            quality_grade=classify_quality(quality_factor, Carrier.COOLING),
            boundary_id=load.boundary_id,
            reference_state_id=load.reference_state_id,
            metadata=load.metadata,
        )

    def battery_state_exergy(self, state: BatteryState, availability_factor: float | None = None) -> float:
        self._require_bound(state.boundary_id, state.reference_state_id)
        availability = state.soh if availability_factor is None else availability_factor
        return battery_stored_exergy(state.stored_energy_j, availability)

    def thermal_storage_state_exergy(self, state: ThermalStorageState) -> float:
        self._require_bound(state.boundary_id, state.reference_state_id)
        return thermal_storage_exergy(state.layers, self.reference_temperature_k)

    def chemical_flow(self, flow: ChemicalFlow) -> ExergyFlow:
        """Return an ExergyFlow for a bound chemical flow.

        The specific chemical exergy [J/kg] must already be set on the flow
        (use the chemical models in eie.chemical to compute it).  The kernel
        verifies boundary and reference-state binding, then converts to a
        typed ExergyFlow with carrier=CHEMICAL.

        Thermodynamic assumption: chemical exergy is independent of the
        reference temperature when computed from the Szargut standard-state
        database.  The reference state binding is retained for accounting
        consistency but does not alter the chemical exergy value.
        """
        import dataclasses

        self._require_bound(flow.boundary_id, flow.reference_state_id)
        exergy_rate = flow.exergy_rate_w
        energy_rate = exergy_rate   # chemical: energy ≈ exergy at Szargut standard state
        quality_factor = 1.0 if energy_rate == 0 else min(1.0, exergy_rate / energy_rate)
        # ExergyFlow requires power-dimension metadata (W); ChemicalFlow carries mass-flow
        # metadata (kg/s), so we create a new Metadata with the correct unit.
        power_metadata = dataclasses.replace(flow.metadata, unit="W")
        return ExergyFlow(
            flow_id=f"{flow.flow_id}:exergy",
            carrier=Carrier.CHEMICAL,
            source_node_id="chemical_source",
            target_node_id="chemical_sink",
            energy_rate_w=energy_rate,
            exergy_rate_w=exergy_rate,
            quality_factor=quality_factor,
            quality_grade=classify_quality(quality_factor, Carrier.CHEMICAL),
            boundary_id=flow.boundary_id,
            reference_state_id=flow.reference_state_id,
            metadata=power_metadata,
        )

    def metadata(
        self,
        *,
        timestamp: datetime,
        source: str,
        method: MeasurementMethod | str,
        unit: str | None = None,
        confidence: float = 1.0,
        uncertainty: float | None = None,
    ) -> Metadata:
        return Metadata(
            timestamp=timestamp,
            source=source,
            method=method,
            unit=unit,
            confidence=confidence,
            uncertainty=uncertainty,
            boundary_id=self.boundary.boundary_id,
            reference_state_id=self.reference_state.reference_state_id,
        )
