from __future__ import annotations

from datetime import timedelta

import pytest

from eie.boundary.boundary import Boundary
from eie.boundary.manager import BoundaryManager
from eie.core.enums import BoundaryType, Carrier, MeasurementMethod, QualityGrade
from eie.core.errors import BoundaryError, DomainError, MissingReferenceError, StaleReferenceError
from eie.exergy.kernel import ExergyKernelV0
from eie.exergy.storage import thermal_storage_exergy, thermal_storage_exergy_breakdown
from eie.flows.base import ExergyFlow, Metadata
from eie.flows.cooling import CoolingLoad
from eie.flows.electrical import ElectricalFlow
from eie.flows.storage import BatteryState, ThermalLayer, ThermalStorageState
from eie.flows.thermal import ThermalFlow
from eie.ledger.loss_fingerprint import LossFingerprint
from eie.reference.state import ReferenceState, utc_now
from eie.reference.validation import (
    require_fresh_reference_state,
    require_reference_state,
    require_reference_state_id,
)


class ObjectLayer:
    def __init__(self, temperature_k: float, energy_j: float) -> None:
        self.temperature_k = temperature_k
        self.energy_j = energy_j


def test_reference_validation_helpers(reference, now):
    assert require_reference_state(reference) == reference
    assert require_fresh_reference_state(reference, at=now) == reference
    assert require_reference_state_id(reference.reference_state_id) == reference.reference_state_id

    with pytest.raises(MissingReferenceError):
        require_reference_state(None)
    with pytest.raises(MissingReferenceError):
        require_reference_state_id("")

    stale = ReferenceState(
        reference_state_id="stale",
        timestamp=now - timedelta(hours=2),
        ambient_temperature_k=300.0,
        ambient_pressure_pa=101_325.0,
        confidence=1.0,
    )
    with pytest.raises(StaleReferenceError):
        require_fresh_reference_state(stale, at=now)


def test_reference_state_validates_optional_fields(now):
    with pytest.raises(DomainError):
        ReferenceState(
            reference_state_id="bad-humidity",
            timestamp=now,
            ambient_temperature_k=300.0,
            ambient_pressure_pa=101_325.0,
            relative_humidity=1.1,
            confidence=1.0,
        )
    with pytest.raises(DomainError):
        ReferenceState(
            reference_state_id="bad-sky",
            timestamp=now,
            ambient_temperature_k=300.0,
            ambient_pressure_pa=101_325.0,
            sky_temperature_k=0.0,
            confidence=1.0,
        )
    with pytest.raises(DomainError):
        ReferenceState(
            reference_state_id="bad-pressure",
            timestamp=now,
            ambient_temperature_k=300.0,
            ambient_pressure_pa=-1.0,
            confidence=1.0,
        )
    with pytest.raises(DomainError):
        ReferenceState(
            reference_state_id="bad-validity",
            timestamp=now,
            ambient_temperature_k=300.0,
            ambient_pressure_pa=101_325.0,
            confidence=1.0,
            valid_until=now - timedelta(seconds=1),
        )
    assert utc_now().tzinfo is not None


def test_boundary_contains_and_manager_listing(reference):
    boundary = Boundary(
        boundary_id="contains-boundary",
        boundary_type=BoundaryType.SITE,
        included_entity_ids=["included"],
        excluded_entity_ids=["excluded"],
        reference_state_id=reference.reference_state_id,
    )
    assert boundary.contains("included")
    assert not boundary.contains("excluded")
    assert not boundary.contains("unknown")

    manager = BoundaryManager()
    manager.add(boundary)
    assert manager.has(boundary.boundary_id)
    assert manager.get(boundary.boundary_id) == boundary
    assert manager.all() == [boundary]
    with pytest.raises(BoundaryError):
        manager.get("missing-boundary")


def test_kernel_exercises_all_flow_methods(kernel, boundary, reference, metadata, energy_metadata, now):
    electrical = ElectricalFlow(
        flow_id="grid-flow",
        real_power_w=100.0,
        voltage_v=230.0,
        frequency_hz=50.0,
        power_factor=0.98,
        thd=0.10,
        availability_factor=0.80,
        boundary_id=boundary.boundary_id,
        reference_state_id=reference.reference_state_id,
        metadata=metadata,
    )
    service_flow = kernel.electrical_flow(electrical, service_derated=True)
    assert service_flow.flags == ["service_derated_not_thermodynamic_loss"]
    assert service_flow.quality_factor == pytest.approx(0.72)

    thermal = ThermalFlow(
        flow_id="hot-water",
        heat_rate_w=1_000.0,
        source_temperature_k=330.0,
        sink_temperature_k=320.0,
        boundary_temperature_k=325.0,
        reference_state_id=reference.reference_state_id,
        boundary_id=boundary.boundary_id,
        metadata=metadata,
    )
    thermal_flow = kernel.thermal_flow(thermal)
    assert thermal_flow.carrier == Carrier.THERMAL
    assert thermal_flow.quality_grade in {QualityGrade.D, QualityGrade.E}

    cooling = CoolingLoad(
        load_id="cold-room",
        cooling_rate_w=1_000.0,
        cold_temperature_k=280.0,
        reference_state_id=reference.reference_state_id,
        boundary_id=boundary.boundary_id,
        metadata=metadata,
    )
    cooling_flow = kernel.cooling_load(cooling)
    assert cooling_flow.carrier == Carrier.COOLING
    assert cooling_flow.exergy_rate_w > 0.0

    battery = BatteryState(
        storage_id="battery",
        stored_energy_j=1_000.0,
        soc=0.5,
        soh=0.9,
        reserve_energy_j=100.0,
        boundary_id=boundary.boundary_id,
        reference_state_id=reference.reference_state_id,
        metadata=energy_metadata,
    )
    assert kernel.battery_state_exergy(battery, availability_factor=0.5) == pytest.approx(500.0)

    storage = ThermalStorageState(
        storage_id="store",
        layers=[ThermalLayer(temperature_k=330.0, energy_j=1_000.0)],
        boundary_id=boundary.boundary_id,
        reference_state_id=reference.reference_state_id,
        metadata=energy_metadata,
    )
    assert kernel.thermal_storage_state_exergy(storage) > 0.0

    generated_metadata = kernel.metadata(
        timestamp=now,
        source="kernel-test",
        method=MeasurementMethod.SIMULATED,
        unit="W",
    )
    assert generated_metadata.boundary_id == boundary.boundary_id


def test_kernel_rejects_missing_constructor_dependencies(reference, boundary):
    with pytest.raises(MissingReferenceError):
        ExergyKernelV0(reference_state=None, boundary=boundary)  # type: ignore[arg-type]
    with pytest.raises(BoundaryError):
        ExergyKernelV0(reference_state=reference, boundary=None)  # type: ignore[arg-type]


def test_exergy_flow_schema_rejects_missing_and_negative_values(metadata, boundary, reference):
    with pytest.raises(DomainError):
        ExergyFlow(
            flow_id="",
            carrier=Carrier.ELECTRIC,
            source_node_id="source",
            target_node_id="sink",
            energy_rate_w=1.0,
            exergy_rate_w=1.0,
            quality_factor=1.0,
            quality_grade=QualityGrade.A,
            boundary_id=boundary.boundary_id,
            reference_state_id=reference.reference_state_id,
            metadata=metadata,
        )
    with pytest.raises(DomainError):
        ExergyFlow(
            flow_id="flow",
            carrier=Carrier.ELECTRIC,
            source_node_id="",
            target_node_id="sink",
            energy_rate_w=1.0,
            exergy_rate_w=1.0,
            quality_factor=1.0,
            quality_grade=QualityGrade.A,
            boundary_id=boundary.boundary_id,
            reference_state_id=reference.reference_state_id,
            metadata=metadata,
        )
    with pytest.raises(DomainError):
        ExergyFlow(
            flow_id="flow",
            carrier=Carrier.ELECTRIC,
            source_node_id="source",
            target_node_id="sink",
            energy_rate_w=-1.0,
            exergy_rate_w=1.0,
            quality_factor=1.0,
            quality_grade=QualityGrade.A,
            boundary_id=boundary.boundary_id,
            reference_state_id=reference.reference_state_id,
            metadata=metadata,
        )


def test_storage_layer_coercion_paths_and_rejection():
    layers = [
        {"temperature_k": 330.0, "energy_j": 1_000.0},
        (320.0, 500.0),
        ObjectLayer(310.0, 250.0),
        ThermalLayer(temperature_k=305.0, energy_j=0.0),
    ]
    breakdown = thermal_storage_exergy_breakdown(layers, 300.0)

    assert breakdown.hot_exergy_j > 0.0
    assert breakdown.cold_exergy_j == 0.0
    assert breakdown.total_exergy_j == pytest.approx(thermal_storage_exergy(layers, 300.0))
    with pytest.raises(TypeError):
        thermal_storage_exergy([object()], 300.0)  # type: ignore[list-item]


def test_loss_fingerprint_validates_all_loss_fields(boundary, reference, now):
    fingerprint = LossFingerprint(
        loss_fingerprint_id="loss-1",
        timestamp=now,
        boundary_id=boundary.boundary_id,
        reference_state_id=reference.reference_state_id,
        destroyed_exergy_j=1.0,
        rejected_recoverable_exergy_j=2.0,
        quality_waste_j=3.0,
        curtailment_j=4.0,
        storage_loss_j=5.0,
        thermal_transfer_loss_j=6.0,
        avoidability_score=0.8,
        causal_hypotheses=["quality mismatch"],
        recommended_actions=["route waste heat to storage"],
    )
    assert fingerprint.avoidability_score == 0.8

    base = {
        "loss_fingerprint_id": "loss-bad",
        "timestamp": now,
        "boundary_id": boundary.boundary_id,
        "reference_state_id": reference.reference_state_id,
        "destroyed_exergy_j": 1.0,
        "rejected_recoverable_exergy_j": 1.0,
        "quality_waste_j": 1.0,
        "curtailment_j": 1.0,
        "storage_loss_j": 1.0,
        "thermal_transfer_loss_j": 1.0,
        "avoidability_score": 0.5,
    }
    for field in [
        "destroyed_exergy_j",
        "rejected_recoverable_exergy_j",
        "quality_waste_j",
        "curtailment_j",
        "storage_loss_j",
        "thermal_transfer_loss_j",
    ]:
        with pytest.raises(DomainError):
            LossFingerprint(**{**base, field: -1.0})
    with pytest.raises(ValueError):
        LossFingerprint(**{**base, "loss_fingerprint_id": ""})
    with pytest.raises(DomainError):
        LossFingerprint(**{**base, "avoidability_score": 1.1})
