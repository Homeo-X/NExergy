from __future__ import annotations

from datetime import timedelta

import pytest

from eie.boundary.boundary import Boundary
from eie.boundary.manager import BoundaryManager
from eie.core.enums import BoundaryType, Carrier, MeasurementMethod, QualityGrade
from eie.core.errors import BoundaryError, DomainError, MissingReferenceError, UnitError
from eie.flows.base import ExergyFlow, Metadata
from eie.flows.chemical import ChemicalFlow
from eie.flows.cooling import CoolingLoad
from eie.flows.electrical import ElectricalFlow
from eie.flows.storage import BatteryState, ThermalLayer, ThermalStorageState
from eie.flows.thermal import ThermalFlow


def test_metadata_rejects_unknown_units(now):
    with pytest.raises(UnitError):
        Metadata(
            timestamp=now,
            source="unit-test",
            method=MeasurementMethod.SIMULATED,
            unit="BTU/hr",
            confidence=1.0,
        )


def test_metadata_rejects_bad_confidence(now):
    with pytest.raises(DomainError):
        Metadata(
            timestamp=now,
            source="unit-test",
            method=MeasurementMethod.SIMULATED,
            unit="W",
            confidence=-0.1,
        )


def test_metadata_require_binding_distinguishes_missing_boundary_and_reference(now):
    missing_boundary = Metadata(
        timestamp=now,
        source="unit-test",
        method=MeasurementMethod.SIMULATED,
        unit="W",
        confidence=1.0,
        boundary_id=None,
        reference_state_id="ref",
    )
    with pytest.raises(BoundaryError):
        missing_boundary.require_binding()

    missing_reference = Metadata(
        timestamp=now,
        source="unit-test",
        method=MeasurementMethod.SIMULATED,
        unit="W",
        confidence=1.0,
        boundary_id="boundary",
        reference_state_id=None,
    )
    with pytest.raises(MissingReferenceError):
        missing_reference.require_binding()


def test_exergy_flow_rejects_metadata_boundary_mismatch(reference, boundary, now):
    bad_metadata = Metadata(
        timestamp=now,
        source="unit-test",
        method=MeasurementMethod.SIMULATED,
        unit="W",
        confidence=1.0,
        boundary_id="other-boundary",
        reference_state_id=reference.reference_state_id,
    )
    with pytest.raises(BoundaryError):
        ExergyFlow(
            flow_id="flow",
            carrier=Carrier.ELECTRIC,
            source_node_id="source",
            target_node_id="sink",
            energy_rate_w=10.0,
            exergy_rate_w=10.0,
            quality_factor=1.0,
            quality_grade=QualityGrade.A,
            boundary_id=boundary.boundary_id,
            reference_state_id=reference.reference_state_id,
            metadata=bad_metadata,
        )


def test_exergy_flow_rejects_metadata_reference_mismatch(reference, boundary, now):
    bad_metadata = Metadata(
        timestamp=now,
        source="unit-test",
        method=MeasurementMethod.SIMULATED,
        unit="W",
        confidence=1.0,
        boundary_id=boundary.boundary_id,
        reference_state_id="other-reference",
    )
    with pytest.raises(MissingReferenceError):
        ExergyFlow(
            flow_id="flow",
            carrier=Carrier.ELECTRIC,
            source_node_id="source",
            target_node_id="sink",
            energy_rate_w=10.0,
            exergy_rate_w=10.0,
            quality_factor=1.0,
            quality_grade=QualityGrade.A,
            boundary_id=boundary.boundary_id,
            reference_state_id=reference.reference_state_id,
            metadata=bad_metadata,
        )


def test_boundary_rejects_included_excluded_overlap(reference, now):
    with pytest.raises(BoundaryError):
        Boundary(
            boundary_id="bad-boundary",
            boundary_type=BoundaryType.SITE,
            included_entity_ids=["a", "b"],
            excluded_entity_ids=["b"],
            reference_state_id=reference.reference_state_id,
            accounting_period_start=now,
            accounting_period_end=now + timedelta(hours=1),
        )


def test_boundary_rejects_backwards_accounting_period(reference, now):
    with pytest.raises(BoundaryError):
        Boundary(
            boundary_id="bad-boundary",
            boundary_type=BoundaryType.SITE,
            included_entity_ids=[],
            excluded_entity_ids=[],
            reference_state_id=reference.reference_state_id,
            accounting_period_start=now,
            accounting_period_end=now - timedelta(seconds=1),
        )


def test_boundary_manager_rejects_duplicates_and_reference_mismatch(boundary):
    manager = BoundaryManager()
    manager.add(boundary)
    with pytest.raises(BoundaryError):
        manager.add(boundary)
    with pytest.raises(BoundaryError):
        manager.require_reference_match(boundary.boundary_id, "other-reference")
    assert manager.require_reference_match(boundary.boundary_id, boundary.reference_state_id) == boundary


def test_typed_flows_reject_invalid_physical_values(boundary, reference, metadata):
    with pytest.raises(DomainError):
        ElectricalFlow(
            flow_id="bad-electrical",
            real_power_w=1.0,
            voltage_v=None,
            frequency_hz=None,
            power_factor=None,
            thd=1.5,
            availability_factor=None,
            boundary_id=boundary.boundary_id,
            reference_state_id=reference.reference_state_id,
            metadata=metadata,
        )

    with pytest.raises(DomainError):
        ThermalFlow(
            flow_id="bad-thermal",
            heat_rate_w=1.0,
            source_temperature_k=0.0,
            sink_temperature_k=None,
            boundary_temperature_k=None,
            reference_state_id=reference.reference_state_id,
            boundary_id=boundary.boundary_id,
            metadata=metadata,
        )

    with pytest.raises(DomainError):
        CoolingLoad(
            load_id="bad-cooling",
            cooling_rate_w=-1.0,
            cold_temperature_k=280.0,
            reference_state_id=reference.reference_state_id,
            boundary_id=boundary.boundary_id,
            metadata=metadata,
        )


def test_storage_schemas_reject_invalid_reserve_and_empty_layers(boundary, reference, metadata):
    with pytest.raises(ValueError):
        BatteryState(
            storage_id="bad-battery",
            stored_energy_j=100.0,
            soc=0.5,
            soh=0.9,
            reserve_energy_j=101.0,
            boundary_id=boundary.boundary_id,
            reference_state_id=reference.reference_state_id,
            metadata=metadata,
        )

    with pytest.raises(ValueError):
        ThermalStorageState(
            storage_id="bad-store",
            layers=[],
            boundary_id=boundary.boundary_id,
            reference_state_id=reference.reference_state_id,
            metadata=metadata,
        )

    with pytest.raises(DomainError):
        ThermalLayer(temperature_k=300.0, energy_j=-1.0)


def test_chemical_flow_is_model_specific_and_computes_exergy_rate(
    boundary,
    reference,
    mass_flow_metadata,
):
    flow = ChemicalFlow(
        flow_id="fuel-flow",
        mass_flow_kg_s=0.1,
        specific_chemical_exergy_j_per_kg=42_000_000.0,
        model_id="simple-fuel-specific-exergy",
        reference_environment_id="ref-env-test",
        boundary_id=boundary.boundary_id,
        reference_state_id=reference.reference_state_id,
        metadata=mass_flow_metadata,
    )
    assert flow.exergy_rate_w == pytest.approx(4_200_000.0)
