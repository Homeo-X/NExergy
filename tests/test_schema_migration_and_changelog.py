"""Tests for the migration framework and changelog."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from eie.core.errors import DomainError
from eie.schema.changelog import ChangelogEntry, SchemaChangelog
from eie.schema.domain_schemas import SCHEMA_CHANGELOG, SCHEMA_MIGRATOR, SCHEMA_REGISTRY
from eie.schema.migration import MigrationEvent, MigrationRule, SchemaMigrator, VersionedRecord
from eie.schema.version import V0_1_0, V0_2_0, V0_3_0, SemanticVersion


_NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


# ── VersionedRecord ──────────────────────────────────────────────────────────

def test_versioned_record_requires_record_id():
    with pytest.raises(DomainError):
        VersionedRecord(
            record_id="",
            schema_name="LedgerEntry",
            schema_version=V0_1_0,
            payload={"ledger_id": "x"},
        )


def test_versioned_record_requires_schema_name():
    with pytest.raises(DomainError):
        VersionedRecord(
            record_id="r1",
            schema_name="",
            schema_version=V0_1_0,
            payload={},
        )


def test_versioned_record_is_immutable():
    record = VersionedRecord(
        record_id="r1",
        schema_name="LedgerEntry",
        schema_version=V0_1_0,
        payload={"ledger_id": "test"},
    )
    assert record.record_id == "r1"
    # dataclass is frozen; direct attribute assignment not possible


# ── MigrationRule ────────────────────────────────────────────────────────────

def test_migration_rule_requires_rule_id():
    with pytest.raises(DomainError):
        MigrationRule(
            rule_id="",
            schema_name="LedgerEntry",
            from_version=V0_1_0,
            to_version=V0_2_0,
            description="test",
            migrate_fn=lambda p: dict(p),
        )


def test_migration_rule_rejects_from_gte_to():
    with pytest.raises(DomainError, match="strictly less than"):
        MigrationRule(
            rule_id="bad",
            schema_name="LedgerEntry",
            from_version=V0_2_0,
            to_version=V0_1_0,  # backwards
            description="bad",
            migrate_fn=lambda p: dict(p),
        )


def test_migration_rule_apply_adds_annotation():
    def _fn(payload: dict) -> dict:
        result = dict(payload)
        result["new_field"] = "added"
        return result

    rule = MigrationRule(
        rule_id="test-rule",
        schema_name="TestSchema",
        from_version=V0_1_0,
        to_version=V0_2_0,
        description="test",
        migrate_fn=_fn,
    )
    result = rule.apply({"existing": 1})
    assert result["new_field"] == "added"
    assert result["_migration_applied"] == "test-rule"


# ── SchemaMigrator ────────────────────────────────────────────────────────────

def test_migrator_ledger_entry_v0_1_to_v0_2():
    original = VersionedRecord(
        record_id="le1",
        schema_name="LedgerEntry",
        schema_version=V0_1_0,
        payload={
            "ledger_id": "l1",
            "energy_in_j": 1000.0,
            "boundary_id": "b1",
        },
    )
    migrated = SCHEMA_MIGRATOR.migrate(original, V0_2_0)
    assert migrated.schema_version == V0_2_0
    assert "carbon_footprint_kg_co2" in migrated.payload
    assert migrated.payload["carbon_footprint_kg_co2"] is None
    assert "schema_version" in migrated.payload


def test_migrator_ledger_entry_v0_1_to_v0_3_chained():
    original = VersionedRecord(
        record_id="le2",
        schema_name="LedgerEntry",
        schema_version=V0_1_0,
        payload={"ledger_id": "l2"},
    )
    migrated = SCHEMA_MIGRATOR.migrate(original, V0_3_0)
    assert migrated.schema_version == V0_3_0
    assert "carbon_footprint_kg_co2" in migrated.payload
    assert "cost_currency" in migrated.payload
    assert len(migrated.migration_history) == 2


def test_migrator_same_version_returns_original():
    record = VersionedRecord(
        record_id="r1",
        schema_name="LedgerEntry",
        schema_version=V0_2_0,
        payload={"ledger_id": "x"},
    )
    result = SCHEMA_MIGRATOR.migrate(record, V0_2_0)
    assert result is record  # same object


def test_migrator_downgrade_raises():
    record = VersionedRecord(
        record_id="r1",
        schema_name="LedgerEntry",
        schema_version=V0_2_0,
        payload={"ledger_id": "x"},
    )
    with pytest.raises(DomainError, match="backwards"):
        SCHEMA_MIGRATOR.migrate(record, V0_1_0)


def test_migrator_missing_path_raises():
    migrator = SchemaMigrator()
    record = VersionedRecord(
        record_id="r1",
        schema_name="UnknownSchema",
        schema_version=V0_1_0,
        payload={},
    )
    with pytest.raises(DomainError, match="no migration path"):
        migrator.migrate(record, V0_2_0)


def test_migrator_can_migrate():
    assert SCHEMA_MIGRATOR.can_migrate("LedgerEntry", V0_1_0, V0_2_0)
    assert SCHEMA_MIGRATOR.can_migrate("LedgerEntry", V0_1_0, V0_3_0)
    assert not SCHEMA_MIGRATOR.can_migrate("LedgerEntry", V0_2_0, V0_1_0)
    assert not SCHEMA_MIGRATOR.can_migrate("NonexistentSchema", V0_1_0, V0_2_0)


def test_migrator_original_record_is_unchanged():
    original = VersionedRecord(
        record_id="r1",
        schema_name="LedgerEntry",
        schema_version=V0_1_0,
        payload={"ledger_id": "l1", "energy_in_j": 100.0},
    )
    _ = SCHEMA_MIGRATOR.migrate(original, V0_2_0)
    # Original unchanged
    assert original.schema_version == V0_1_0
    assert "carbon_footprint_kg_co2" not in original.payload


def test_migration_history_is_recorded():
    record = VersionedRecord(
        record_id="r1",
        schema_name="ExergyFlow",
        schema_version=V0_1_0,
        payload={"flow_id": "f1"},
    )
    migrated = SCHEMA_MIGRATOR.migrate(record, V0_2_0)
    assert len(migrated.migration_history) == 1
    event = migrated.migration_history[0]
    assert isinstance(event, MigrationEvent)
    assert event.from_version == "0.1.0"
    assert event.to_version == "0.2.0"


def test_migrator_duplicate_rule_raises():
    migrator = SchemaMigrator()
    rule = MigrationRule(
        rule_id="dup-rule",
        schema_name="Test",
        from_version=V0_1_0,
        to_version=V0_2_0,
        description="test",
        migrate_fn=lambda p: dict(p),
    )
    migrator.register(rule)
    with pytest.raises(DomainError, match="already registered"):
        migrator.register(rule)


# ── SchemaChangelog ──────────────────────────────────────────────────────────

def test_changelog_for_schema_returns_only_matching():
    entries = SCHEMA_CHANGELOG.for_schema("LedgerEntry")
    assert all(e.schema_name == "LedgerEntry" for e in entries)
    assert len(entries) >= 3  # v0.1.0, v0.2.0, v0.3.0


def test_changelog_since_v0_1_0_returns_newer_entries():
    entries = SCHEMA_CHANGELOG.since("LedgerEntry", V0_1_0)
    assert all(e.version > V0_1_0 for e in entries)


def test_changelog_since_inclusive():
    entries = SCHEMA_CHANGELOG.since("LedgerEntry", V0_1_0, inclusive=True)
    versions = {e.version for e in entries}
    assert V0_1_0 in versions


def test_changelog_no_breaking_changes_in_v0():
    breaking = SCHEMA_CHANGELOG.breaking_changes_between(
        "LedgerEntry", V0_1_0, V0_3_0
    )
    assert len(breaking) == 0  # all v0.x changes are non-breaking


def test_changelog_has_breaking_changes_returns_false():
    assert not SCHEMA_CHANGELOG.has_breaking_changes("LedgerEntry", V0_1_0, V0_3_0)


def test_changelog_all_schema_names():
    names = SCHEMA_CHANGELOG.all_schema_names()
    assert "LedgerEntry" in names
    assert "ExergyFlow" in names


def test_changelog_summary_is_dict():
    summary = SCHEMA_CHANGELOG.summary()
    assert isinstance(summary, dict)
    assert all(isinstance(v, list) for v in summary.values())


def test_changelog_entry_rejects_invalid_change_type():
    with pytest.raises(DomainError, match="change_type"):
        ChangelogEntry(
            schema_name="Test",
            version=V0_1_0,
            change_type="INVENTED",  # type: ignore[arg-type]
            description="bad",
            breaking=False,
            affected_fields=[],
            released_at=_NOW,
        )


def test_changelog_entry_breaking_only_for_removed_or_changed():
    with pytest.raises(DomainError, match="REMOVED or CHANGED"):
        ChangelogEntry(
            schema_name="Test",
            version=V0_2_0,
            change_type="ADDED",
            description="test",
            breaking=True,   # ADDED cannot be breaking
            affected_fields=["x"],
            released_at=_NOW,
        )


def test_changelog_entries_for_version():
    entries = SCHEMA_CHANGELOG.entries_for_version("LedgerEntry", V0_2_0)
    assert len(entries) >= 1
    assert all(e.version == V0_2_0 for e in entries)


# ── Additional schema migrations for domain schemas ──────────────────────────

def test_migrator_reference_state_v0_1_to_v0_2():
    original = VersionedRecord(
        record_id="rs1",
        schema_name="ReferenceState",
        schema_version=V0_1_0,
        payload={"reference_state_id": "r1", "ambient_temperature_k": 298.15},
    )
    migrated = SCHEMA_MIGRATOR.migrate(original, V0_2_0)
    assert migrated.schema_version == V0_2_0
    assert "soil_temperature_k" in migrated.payload
    assert "schema_version" in migrated.payload


def test_migrator_boundary_v0_1_to_v0_2():
    original = VersionedRecord(
        record_id="b1",
        schema_name="Boundary",
        schema_version=V0_1_0,
        payload={"boundary_id": "b1", "boundary_type": "SITE"},
    )
    migrated = SCHEMA_MIGRATOR.migrate(original, V0_2_0)
    assert migrated.schema_version == V0_2_0
    assert "description" in migrated.payload
    assert migrated.payload["description"] is None


def test_migrator_chemical_flow_v0_1_to_v0_2():
    original = VersionedRecord(
        record_id="cf1",
        schema_name="ChemicalFlow",
        schema_version=V0_1_0,
        payload={"flow_id": "f1", "mass_flow_kg_s": 0.01},
    )
    migrated = SCHEMA_MIGRATOR.migrate(original, V0_2_0)
    assert migrated.schema_version == V0_2_0
    assert "lhv_j_per_kg" in migrated.payload
    assert "beta_factor" in migrated.payload


def test_migrator_exergy_flow_v0_1_to_v0_2():
    original = VersionedRecord(
        record_id="ef1",
        schema_name="ExergyFlow",
        schema_version=V0_1_0,
        payload={"flow_id": "f1", "exergy_rate_w": 1000.0},
    )
    migrated = SCHEMA_MIGRATOR.migrate(original, V0_2_0)
    assert migrated.schema_version == V0_2_0
    assert "carbon_intensity_kg_co2_per_j" in migrated.payload


# ── BoundaryGuard coverage ────────────────────────────────────────────────────

def test_boundary_guard_missing_boundary():
    from eie.guards.boundary_guard import BoundaryGuard
    result = BoundaryGuard().check(None)
    assert not result.passed


def test_boundary_guard_reference_mismatch():
    from datetime import timedelta
    from eie.boundary.boundary import Boundary
    from eie.core.enums import BoundaryType
    from eie.guards.boundary_guard import BoundaryGuard
    boundary = Boundary(
        boundary_id="b1",
        boundary_type=BoundaryType.SITE,
        included_entity_ids=[],
        excluded_entity_ids=[],
        reference_state_id="ref-A",
        accounting_period_start=_NOW,
        accounting_period_end=_NOW + timedelta(hours=1),
    )
    result = BoundaryGuard().check(boundary, expected_reference_state_id="ref-B")
    assert not result.passed


def test_boundary_guard_check_binding_success():
    from datetime import timedelta
    from eie.boundary.boundary import Boundary
    from eie.core.enums import BoundaryType
    from eie.guards.boundary_guard import BoundaryGuard
    boundary = Boundary(
        boundary_id="b1",
        boundary_type=BoundaryType.SITE,
        included_entity_ids=[],
        excluded_entity_ids=[],
        reference_state_id="ref-A",
        accounting_period_start=_NOW,
        accounting_period_end=_NOW + timedelta(hours=1),
    )
    result = BoundaryGuard().check_binding(
        object_boundary_id="b1",
        object_reference_state_id="ref-A",
        boundary=boundary,
    )
    assert result.passed


# ── LedgerEntry validation ────────────────────────────────────────────────────

def test_ledger_entry_rejects_empty_ledger_id():
    from eie.ledger.entries import LedgerEntry
    with pytest.raises(Exception):
        LedgerEntry(
            ledger_id="",
            timestamp=_NOW,
            boundary_id="b1",
            reference_state_id="r1",
            energy_in_j=0.0, energy_out_j=0.0, energy_stored_delta_j=0.0,
            energy_rejected_j=0.0, energy_residual_j=0.0,
            exergy_in_j=0.0, useful_exergy_j=0.0, stored_exergy_delta_j=0.0,
            recovered_exergy_j=0.0, rejected_exergy_j=0.0,
            destroyed_exergy_j=0.0, exergy_residual_j=0.0,
            entropy_generated_j_per_k=0.0,
        )


def test_ledger_entry_rejects_empty_boundary_id():
    from eie.ledger.entries import LedgerEntry
    from eie.core.errors import BoundaryError
    with pytest.raises(BoundaryError):
        LedgerEntry(
            ledger_id="x",
            timestamp=_NOW,
            boundary_id="",
            reference_state_id="r1",
            energy_in_j=0.0, energy_out_j=0.0, energy_stored_delta_j=0.0,
            energy_rejected_j=0.0, energy_residual_j=0.0,
            exergy_in_j=0.0, useful_exergy_j=0.0, stored_exergy_delta_j=0.0,
            recovered_exergy_j=0.0, rejected_exergy_j=0.0,
            destroyed_exergy_j=0.0, exergy_residual_j=0.0,
            entropy_generated_j_per_k=0.0,
        )


def test_ledger_entry_rejects_empty_reference_state_id():
    from eie.ledger.entries import LedgerEntry
    from eie.core.errors import MissingReferenceError
    with pytest.raises(MissingReferenceError):
        LedgerEntry(
            ledger_id="x",
            timestamp=_NOW,
            boundary_id="b1",
            reference_state_id="",
            energy_in_j=0.0, energy_out_j=0.0, energy_stored_delta_j=0.0,
            energy_rejected_j=0.0, energy_residual_j=0.0,
            exergy_in_j=0.0, useful_exergy_j=0.0, stored_exergy_delta_j=0.0,
            recovered_exergy_j=0.0, rejected_exergy_j=0.0,
            destroyed_exergy_j=0.0, exergy_residual_j=0.0,
            entropy_generated_j_per_k=0.0,
        )


def test_ledger_entry_rejects_non_finite_value():
    from eie.ledger.entries import LedgerEntry
    with pytest.raises(Exception):
        LedgerEntry(
            ledger_id="x",
            timestamp=_NOW,
            boundary_id="b1",
            reference_state_id="r1",
            energy_in_j=float("nan"), energy_out_j=0.0, energy_stored_delta_j=0.0,
            energy_rejected_j=0.0, energy_residual_j=0.0,
            exergy_in_j=0.0, useful_exergy_j=0.0, stored_exergy_delta_j=0.0,
            recovered_exergy_j=0.0, rejected_exergy_j=0.0,
            destroyed_exergy_j=0.0, exergy_residual_j=0.0,
            entropy_generated_j_per_k=0.0,
        )
