"""Tests for schema declarations and the schema registry."""

from __future__ import annotations

import pytest

from eie.core.errors import DomainError
from eie.schema.declaration import FieldDeclaration, SchemaDeclaration
from eie.schema.domain_schemas import (
    LEDGER_ENTRY_V0_1_0,
    LEDGER_ENTRY_V0_2_0,
    EXERGY_FLOW_V0_1_0,
    SCHEMA_REGISTRY,
)
from eie.schema.registry import SchemaRegistry
from eie.schema.version import V0_1_0, V0_2_0, SemanticVersion


# ── FieldDeclaration ─────────────────────────────────────────────────────────

def test_field_declaration_requires_name():
    with pytest.raises(DomainError):
        FieldDeclaration(name="", type_hint="str", required=True, added_in=V0_1_0)


def test_field_declaration_requires_type_hint():
    with pytest.raises(DomainError):
        FieldDeclaration(name="x", type_hint="", required=True, added_in=V0_1_0)


def test_field_declaration_removed_before_deprecated_raises():
    with pytest.raises(DomainError, match="removed_in.*before"):
        FieldDeclaration(
            name="x",
            type_hint="str",
            required=False,
            added_in=V0_1_0,
            deprecated_in=V0_2_0,
            removed_in=V0_1_0,  # before deprecated_in
        )


def test_field_is_present_in_correct_versions():
    fd = FieldDeclaration(name="x", type_hint="str", required=True, added_in=V0_2_0)
    assert not fd.is_present_in(V0_1_0)
    assert fd.is_present_in(V0_2_0)


def test_field_is_deprecated_in_correct_version():
    fd = FieldDeclaration(
        name="old_field", type_hint="str", required=False,
        added_in=V0_1_0, deprecated_in=V0_2_0
    )
    assert not fd.is_deprecated_in(V0_1_0)
    assert fd.is_deprecated_in(V0_2_0)


def test_field_not_present_after_removal():
    v0_3 = SemanticVersion(0, 3, 0)
    fd = FieldDeclaration(
        name="removed_field", type_hint="str", required=False,
        added_in=V0_1_0, deprecated_in=V0_2_0, removed_in=v0_3
    )
    assert fd.is_present_in(V0_2_0)
    assert not fd.is_present_in(v0_3)


# ── SchemaDeclaration ────────────────────────────────────────────────────────

def test_schema_declaration_requires_name():
    with pytest.raises(DomainError):
        SchemaDeclaration(schema_name="", schema_version=V0_1_0, description="x")


def test_schema_declaration_requires_description():
    with pytest.raises(DomainError):
        SchemaDeclaration(schema_name="X", schema_version=V0_1_0, description="")


def test_schema_declaration_rejects_duplicate_field_names():
    fd = FieldDeclaration(name="x", type_hint="str", required=True, added_in=V0_1_0)
    with pytest.raises(DomainError, match="duplicate field"):
        SchemaDeclaration(schema_name="Bad", schema_version=V0_1_0, description="bad", fields=[fd, fd])


def test_ledger_entry_v0_1_0_has_all_required_fields():
    decl = LEDGER_ENTRY_V0_1_0
    required_names = {f.name for f in decl.required_fields()}
    for name in ["ledger_id", "timestamp", "boundary_id", "reference_state_id",
                 "energy_in_j", "exergy_in_j", "destroyed_exergy_j", "entropy_generated_j_per_k"]:
        assert name in required_names


def test_ledger_entry_v0_2_0_has_new_optional_field():
    decl = LEDGER_ENTRY_V0_2_0
    optional_names = {f.name for f in decl.optional_fields()}
    assert "carbon_footprint_kg_co2" in optional_names
    assert "schema_version" in optional_names


def test_schema_validate_record_no_errors_for_valid_data():
    valid = {
        "ledger_id": "test-1",
        "timestamp": "2026-01-01T12:00:00Z",
        "boundary_id": "b1",
        "reference_state_id": "r1",
        "energy_in_j": 1000.0,
        "energy_out_j": 800.0,
        "energy_stored_delta_j": 0.0,
        "energy_rejected_j": 200.0,
        "energy_residual_j": 0.0,
        "exergy_in_j": 1000.0,
        "useful_exergy_j": 700.0,
        "stored_exergy_delta_j": 0.0,
        "recovered_exergy_j": 0.0,
        "rejected_exergy_j": 100.0,
        "destroyed_exergy_j": 200.0,
        "exergy_residual_j": 0.0,
        "entropy_generated_j_per_k": 0.67,
    }
    errors = LEDGER_ENTRY_V0_1_0.validate_record(valid)
    assert errors == []


def test_schema_validate_record_detects_missing_required_field():
    incomplete = {"ledger_id": "x"}  # missing many required fields
    errors = LEDGER_ENTRY_V0_1_0.validate_record(incomplete)
    assert any("timestamp" in e for e in errors)


def test_schema_validate_record_detects_unexpected_field():
    record = {"ledger_id": "x", "nonexistent_field": 42}
    errors = LEDGER_ENTRY_V0_1_0.validate_record(record)
    assert any("nonexistent_field" in e for e in errors)


# ── SchemaRegistry ────────────────────────────────────────────────────────────

def test_registry_contains_all_expected_schemas():
    schemas = SCHEMA_REGISTRY.known_schemas()
    for name in ["LedgerEntry", "ExergyFlow", "ReferenceState", "Boundary",
                 "ChemicalFlow", "BatteryState", "ThermalStorageState"]:
        assert name in schemas


def test_registry_current_version_is_highest():
    current = SCHEMA_REGISTRY.current_version("LedgerEntry")
    all_v = [d.schema_version for d in SCHEMA_REGISTRY.all_versions("LedgerEntry")]
    assert current == max(all_v)


def test_registry_get_exact_version():
    decl = SCHEMA_REGISTRY.get("LedgerEntry", V0_1_0)
    assert decl.schema_name == "LedgerEntry"
    assert decl.schema_version == V0_1_0


def test_registry_get_unknown_version_raises():
    unknown = SemanticVersion(9, 9, 9)
    with pytest.raises(DomainError, match="no schema"):
        SCHEMA_REGISTRY.get("LedgerEntry", unknown)


def test_registry_get_unknown_schema_raises():
    with pytest.raises(DomainError, match="no schema"):
        SCHEMA_REGISTRY.get("Nonexistent", V0_1_0)


def test_registry_is_registered():
    assert SCHEMA_REGISTRY.is_registered("LedgerEntry")
    assert SCHEMA_REGISTRY.is_registered("LedgerEntry", V0_1_0)
    assert not SCHEMA_REGISTRY.is_registered("LedgerEntry", SemanticVersion(9, 9, 9))
    assert not SCHEMA_REGISTRY.is_registered("DoesNotExist")


def test_registry_all_versions_sorted():
    versions = [d.schema_version for d in SCHEMA_REGISTRY.all_versions("LedgerEntry")]
    assert versions == sorted(versions)


def test_registry_duplicate_registration_raises():
    registry = SchemaRegistry()
    registry.register(LEDGER_ENTRY_V0_1_0)
    with pytest.raises(DomainError, match="already registered"):
        registry.register(LEDGER_ENTRY_V0_1_0)


def test_registry_compatible_version_finds_older():
    compatible = SCHEMA_REGISTRY.compatible_version("LedgerEntry", V0_2_0)
    assert V0_1_0 in compatible
    assert V0_2_0 in compatible


def test_registry_summary_returns_dict():
    summary = SCHEMA_REGISTRY.summary()
    assert isinstance(summary, dict)
    assert "LedgerEntry" in summary


def test_registry_is_field_present_in():
    assert SCHEMA_REGISTRY.is_field_present_in("LedgerEntry", "ledger_id", V0_1_0)
    assert SCHEMA_REGISTRY.is_field_present_in("LedgerEntry", "carbon_footprint_kg_co2", V0_2_0)
    assert not SCHEMA_REGISTRY.is_field_present_in("LedgerEntry", "carbon_footprint_kg_co2", V0_1_0)
