"""Domain schema declarations and migrations for all EIE v0 types.

This module is the single source of truth for:
1. Schema declarations for all domain objects (current and historical versions)
2. Migration rules between versions
3. The changelog entries for each schema change
4. The populated SchemaRegistry, SchemaMigrator, and SchemaChangelog singletons

After import, use:
    from eie.schema.domain_schemas import (
        SCHEMA_REGISTRY, SCHEMA_MIGRATOR, SCHEMA_CHANGELOG
    )

Schema versioning
─────────────────
v0.1.0 — initial release for all schemas
v0.2.0 — non-breaking additions (optional fields) for:
            LedgerEntry, ExergyFlow, ReferenceState, Boundary
v0.3.0 — schema_version field added to all schemas (non-breaking, optional)

Note: all v0.x changes are non-breaking (minor/patch bumps only).
The first breaking change (v1.0.0) will require a major-version bump.
"""

from __future__ import annotations

from datetime import datetime, timezone

from eie.schema.changelog import ChangelogEntry, SchemaChangelog
from eie.schema.declaration import FieldDeclaration, SchemaDeclaration
from eie.schema.migration import MigrationRule, SchemaMigrator, VersionedRecord
from eie.schema.registry import SchemaRegistry
from eie.schema.version import V0_1_0, V0_2_0, V0_3_0, SemanticVersion

_RELEASE_DATE_V0_1_0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_RELEASE_DATE_V0_2_0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
_RELEASE_DATE_V0_3_0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helper: common binding fields (present in every bound domain object)
# ---------------------------------------------------------------------------

def _binding_fields(added_in: SemanticVersion) -> list[FieldDeclaration]:
    return [
        FieldDeclaration(
            name="boundary_id",
            type_hint="str",
            required=True,
            added_in=added_in,
            notes="Accounting boundary that this record belongs to",
        ),
        FieldDeclaration(
            name="reference_state_id",
            type_hint="str",
            required=True,
            added_in=added_in,
            notes="Reference state used for exergy accounting",
        ),
    ]


def _schema_version_field(added_in: SemanticVersion) -> FieldDeclaration:
    return FieldDeclaration(
        name="schema_version",
        type_hint="str",
        required=False,
        default=str(V0_1_0),
        added_in=added_in,
        notes="Schema version string (MAJOR.MINOR.PATCH); absent in older records defaults to 0.1.0",
    )


# ---------------------------------------------------------------------------
# LedgerEntry schema declarations
# ---------------------------------------------------------------------------

_LEDGER_ENTRY_FIELDS_V0_1_0 = [
    FieldDeclaration("ledger_id",              "str",      True,  V0_1_0),
    FieldDeclaration("timestamp",             "datetime",  True,  V0_1_0),
    *_binding_fields(V0_1_0),
    FieldDeclaration("energy_in_j",           "float",     True,  V0_1_0),
    FieldDeclaration("energy_out_j",          "float",     True,  V0_1_0),
    FieldDeclaration("energy_stored_delta_j", "float",     True,  V0_1_0),
    FieldDeclaration("energy_rejected_j",     "float",     True,  V0_1_0),
    FieldDeclaration("energy_residual_j",     "float",     True,  V0_1_0),
    FieldDeclaration("exergy_in_j",           "float",     True,  V0_1_0),
    FieldDeclaration("useful_exergy_j",       "float",     True,  V0_1_0),
    FieldDeclaration("stored_exergy_delta_j", "float",     True,  V0_1_0),
    FieldDeclaration("recovered_exergy_j",    "float",     True,  V0_1_0),
    FieldDeclaration("rejected_exergy_j",     "float",     True,  V0_1_0),
    FieldDeclaration("destroyed_exergy_j",    "float",     True,  V0_1_0),
    FieldDeclaration("exergy_residual_j",     "float",     True,  V0_1_0),
    FieldDeclaration("entropy_generated_j_per_k", "float", True,  V0_1_0),
    FieldDeclaration("flags",                 "list[str]", False, V0_1_0, default=[]),
    FieldDeclaration("confidence",            "float",     False, V0_1_0, default=1.0),
]

LEDGER_ENTRY_V0_1_0 = SchemaDeclaration(
    schema_name="LedgerEntry",
    schema_version=V0_1_0,
    description="Append-only energy/exergy ledger entry for one accounting period",
    fields=_LEDGER_ENTRY_FIELDS_V0_1_0,
)

LEDGER_ENTRY_V0_2_0 = SchemaDeclaration(
    schema_name="LedgerEntry",
    schema_version=V0_2_0,
    description="LedgerEntry v0.2.0: adds optional carbon_footprint_kg_co2 and schema_version",
    fields=_LEDGER_ENTRY_FIELDS_V0_1_0 + [
        FieldDeclaration(
            "carbon_footprint_kg_co2",
            "float | None",
            False,
            V0_2_0,
            default=None,
            notes="Optional carbon footprint attributed to this period's energy inputs",
        ),
        _schema_version_field(V0_2_0),
    ],
)

LEDGER_ENTRY_V0_3_0 = SchemaDeclaration(
    schema_name="LedgerEntry",
    schema_version=V0_3_0,
    description="LedgerEntry v0.3.0: adds optional cost_currency field",
    fields=list(LEDGER_ENTRY_V0_2_0.fields) + [
        FieldDeclaration(
            "cost_currency",
            "float | None",
            False,
            V0_3_0,
            default=None,
            notes="Optional marginal operating cost for this period's energy inputs",
        ),
    ],
)

# ---------------------------------------------------------------------------
# ExergyFlow schema declarations
# ---------------------------------------------------------------------------

_EXERGY_FLOW_FIELDS_V0_1_0 = [
    FieldDeclaration("flow_id",        "str",    True,  V0_1_0),
    FieldDeclaration("carrier",        "str",    True,  V0_1_0),
    FieldDeclaration("source_node_id", "str",    True,  V0_1_0),
    FieldDeclaration("target_node_id", "str",    True,  V0_1_0),
    FieldDeclaration("energy_rate_w",  "float",  True,  V0_1_0),
    FieldDeclaration("exergy_rate_w",  "float",  True,  V0_1_0),
    FieldDeclaration("quality_factor", "float",  True,  V0_1_0),
    FieldDeclaration("quality_grade",  "str",    True,  V0_1_0),
    *_binding_fields(V0_1_0),
    FieldDeclaration("flags",          "list[str]", False, V0_1_0, default=[]),
]

EXERGY_FLOW_V0_1_0 = SchemaDeclaration(
    schema_name="ExergyFlow",
    schema_version=V0_1_0,
    description="Typed, boundary-bound, reference-bound exergy flow",
    fields=_EXERGY_FLOW_FIELDS_V0_1_0,
)

EXERGY_FLOW_V0_2_0 = SchemaDeclaration(
    schema_name="ExergyFlow",
    schema_version=V0_2_0,
    description="ExergyFlow v0.2.0: adds optional carbon_intensity_kg_co2_per_j",
    fields=_EXERGY_FLOW_FIELDS_V0_1_0 + [
        FieldDeclaration(
            "carbon_intensity_kg_co2_per_j",
            "float | None",
            False,
            V0_2_0,
            default=None,
            notes="Marginal carbon intensity of the energy carrier at this flow point",
        ),
        _schema_version_field(V0_2_0),
    ],
)

# ---------------------------------------------------------------------------
# ReferenceState schema declarations
# ---------------------------------------------------------------------------

_REFERENCE_STATE_FIELDS_V0_1_0 = [
    FieldDeclaration("reference_state_id",     "str",           True,  V0_1_0),
    FieldDeclaration("timestamp",              "datetime",      True,  V0_1_0),
    FieldDeclaration("ambient_temperature_k",  "float",         True,  V0_1_0),
    FieldDeclaration("ambient_pressure_pa",    "float",         True,  V0_1_0),
    FieldDeclaration("relative_humidity",      "float | None",  False, V0_1_0, default=None),
    FieldDeclaration("sky_temperature_k",      "float | None",  False, V0_1_0, default=None),
    FieldDeclaration("nominal_grid_voltage_v", "float | None",  False, V0_1_0, default=None),
    FieldDeclaration("nominal_grid_frequency_hz", "float | None", False, V0_1_0, default=None),
    FieldDeclaration("marginal_carbon_kg_per_kwh", "float | None", False, V0_1_0, default=None),
    FieldDeclaration("marginal_price_per_kwh", "float | None",  False, V0_1_0, default=None),
    FieldDeclaration("confidence",             "float",         False, V0_1_0, default=1.0),
    FieldDeclaration("valid_until",            "datetime | None", False, V0_1_0, default=None),
    FieldDeclaration("notes",                  "str | None",    False, V0_1_0, default=None),
]

REFERENCE_STATE_V0_1_0 = SchemaDeclaration(
    schema_name="ReferenceState",
    schema_version=V0_1_0,
    description="Dynamic reference environment for exergy accounting",
    fields=_REFERENCE_STATE_FIELDS_V0_1_0,
)

REFERENCE_STATE_V0_2_0 = SchemaDeclaration(
    schema_name="ReferenceState",
    schema_version=V0_2_0,
    description="ReferenceState v0.2.0: adds optional soil_temperature_k and schema_version",
    fields=_REFERENCE_STATE_FIELDS_V0_1_0 + [
        FieldDeclaration(
            "soil_temperature_k",
            "float | None",
            False,
            V0_2_0,
            default=None,
            notes="Near-surface soil temperature for ground-source heat pump exergy accounting",
        ),
        _schema_version_field(V0_2_0),
    ],
)

# ---------------------------------------------------------------------------
# Boundary schema declarations
# ---------------------------------------------------------------------------

_BOUNDARY_FIELDS_V0_1_0 = [
    FieldDeclaration("boundary_id",             "str",           True,  V0_1_0),
    FieldDeclaration("boundary_type",           "str",           True,  V0_1_0),
    FieldDeclaration("included_entity_ids",     "list[str]",     False, V0_1_0, default=[]),
    FieldDeclaration("excluded_entity_ids",     "list[str]",     False, V0_1_0, default=[]),
    FieldDeclaration("reference_state_id",      "str",           True,  V0_1_0),
    FieldDeclaration("accounting_period_start", "datetime | None", False, V0_1_0, default=None),
    FieldDeclaration("accounting_period_end",   "datetime | None", False, V0_1_0, default=None),
]

BOUNDARY_V0_1_0 = SchemaDeclaration(
    schema_name="Boundary",
    schema_version=V0_1_0,
    description="Explicit accounting boundary for exergy flows and ledger entries",
    fields=_BOUNDARY_FIELDS_V0_1_0,
)

BOUNDARY_V0_2_0 = SchemaDeclaration(
    schema_name="Boundary",
    schema_version=V0_2_0,
    description="Boundary v0.2.0: adds optional description and schema_version",
    fields=_BOUNDARY_FIELDS_V0_1_0 + [
        FieldDeclaration(
            "description",
            "str | None",
            False,
            V0_2_0,
            default=None,
            notes="Human-readable description of what this boundary encompasses",
        ),
        _schema_version_field(V0_2_0),
    ],
)

# ---------------------------------------------------------------------------
# ChemicalFlow schema declarations
# ---------------------------------------------------------------------------

CHEMICAL_FLOW_V0_1_0 = SchemaDeclaration(
    schema_name="ChemicalFlow",
    schema_version=V0_1_0,
    description="Model-bound chemical exergy flow with explicit reference environment",
    fields=[
        FieldDeclaration("flow_id",                        "str",   True,  V0_1_0),
        FieldDeclaration("mass_flow_kg_s",                 "float", True,  V0_1_0),
        FieldDeclaration("specific_chemical_exergy_j_per_kg", "float", True, V0_1_0),
        FieldDeclaration("model_id",                       "str",   True,  V0_1_0),
        FieldDeclaration("reference_environment_id",       "str",   True,  V0_1_0),
        *_binding_fields(V0_1_0),
    ],
)

CHEMICAL_FLOW_V0_2_0 = SchemaDeclaration(
    schema_name="ChemicalFlow",
    schema_version=V0_2_0,
    description="ChemicalFlow v0.2.0: adds optional lhv_j_per_kg and beta_factor",
    fields=list(CHEMICAL_FLOW_V0_1_0.fields) + [
        FieldDeclaration(
            "lhv_j_per_kg",
            "float | None",
            False,
            V0_2_0,
            default=None,
            notes="Lower heating value used for beta-factor validation (optional)",
        ),
        FieldDeclaration(
            "beta_factor",
            "float | None",
            False,
            V0_2_0,
            default=None,
            notes="Szargut beta factor (exergy/LHV) for documentation",
        ),
        _schema_version_field(V0_2_0),
    ],
)

# ---------------------------------------------------------------------------
# BatteryState and ThermalStorageState schema declarations
# ---------------------------------------------------------------------------

BATTERY_STATE_V0_1_0 = SchemaDeclaration(
    schema_name="BatteryState",
    schema_version=V0_1_0,
    description="Battery state of charge and health for exergy accounting",
    fields=[
        FieldDeclaration("storage_id",       "str",   True,  V0_1_0),
        FieldDeclaration("stored_energy_j",  "float", True,  V0_1_0),
        FieldDeclaration("soc",              "float", True,  V0_1_0),
        FieldDeclaration("soh",              "float", True,  V0_1_0),
        FieldDeclaration("reserve_energy_j", "float", True,  V0_1_0),
        *_binding_fields(V0_1_0),
    ],
)

THERMAL_STORAGE_STATE_V0_1_0 = SchemaDeclaration(
    schema_name="ThermalStorageState",
    schema_version=V0_1_0,
    description="Stratified thermal storage state with per-layer temperature and energy",
    fields=[
        FieldDeclaration("storage_id", "str",            True,  V0_1_0),
        FieldDeclaration("layers",     "list[ThermalLayer]", True, V0_1_0),
        *_binding_fields(V0_1_0),
    ],
)

# ---------------------------------------------------------------------------
# Populate registry, migrator, and changelog
# ---------------------------------------------------------------------------

SCHEMA_REGISTRY = SchemaRegistry()
SCHEMA_MIGRATOR = SchemaMigrator()
SCHEMA_CHANGELOG = SchemaChangelog()

# Register all declarations
for _decl in [
    LEDGER_ENTRY_V0_1_0, LEDGER_ENTRY_V0_2_0, LEDGER_ENTRY_V0_3_0,
    EXERGY_FLOW_V0_1_0, EXERGY_FLOW_V0_2_0,
    REFERENCE_STATE_V0_1_0, REFERENCE_STATE_V0_2_0,
    BOUNDARY_V0_1_0, BOUNDARY_V0_2_0,
    CHEMICAL_FLOW_V0_1_0, CHEMICAL_FLOW_V0_2_0,
    BATTERY_STATE_V0_1_0,
    THERMAL_STORAGE_STATE_V0_1_0,
]:
    SCHEMA_REGISTRY.register(_decl)


# ---------------------------------------------------------------------------
# Migration rules
# ---------------------------------------------------------------------------

def _ledger_entry_v0_1_to_v0_2(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.setdefault("carbon_footprint_kg_co2", None)
    result.setdefault("schema_version", str(V0_2_0))
    return result


def _ledger_entry_v0_2_to_v0_3(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.setdefault("cost_currency", None)
    result["schema_version"] = str(V0_3_0)
    return result


def _exergy_flow_v0_1_to_v0_2(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.setdefault("carbon_intensity_kg_co2_per_j", None)
    result.setdefault("schema_version", str(V0_2_0))
    return result


def _reference_state_v0_1_to_v0_2(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.setdefault("soil_temperature_k", None)
    result.setdefault("schema_version", str(V0_2_0))
    return result


def _boundary_v0_1_to_v0_2(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.setdefault("description", None)
    result.setdefault("schema_version", str(V0_2_0))
    return result


def _chemical_flow_v0_1_to_v0_2(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.setdefault("lhv_j_per_kg", None)
    result.setdefault("beta_factor", None)
    result.setdefault("schema_version", str(V0_2_0))
    return result


for _rule in [
    MigrationRule(
        rule_id="LedgerEntry_v0.1.0_to_v0.2.0",
        schema_name="LedgerEntry",
        from_version=V0_1_0,
        to_version=V0_2_0,
        description="Add optional carbon_footprint_kg_co2 and schema_version fields",
        migrate_fn=_ledger_entry_v0_1_to_v0_2,
    ),
    MigrationRule(
        rule_id="LedgerEntry_v0.2.0_to_v0.3.0",
        schema_name="LedgerEntry",
        from_version=V0_2_0,
        to_version=V0_3_0,
        description="Add optional cost_currency field",
        migrate_fn=_ledger_entry_v0_2_to_v0_3,
    ),
    MigrationRule(
        rule_id="ExergyFlow_v0.1.0_to_v0.2.0",
        schema_name="ExergyFlow",
        from_version=V0_1_0,
        to_version=V0_2_0,
        description="Add optional carbon_intensity_kg_co2_per_j and schema_version",
        migrate_fn=_exergy_flow_v0_1_to_v0_2,
    ),
    MigrationRule(
        rule_id="ReferenceState_v0.1.0_to_v0.2.0",
        schema_name="ReferenceState",
        from_version=V0_1_0,
        to_version=V0_2_0,
        description="Add optional soil_temperature_k and schema_version",
        migrate_fn=_reference_state_v0_1_to_v0_2,
    ),
    MigrationRule(
        rule_id="Boundary_v0.1.0_to_v0.2.0",
        schema_name="Boundary",
        from_version=V0_1_0,
        to_version=V0_2_0,
        description="Add optional description and schema_version",
        migrate_fn=_boundary_v0_1_to_v0_2,
    ),
    MigrationRule(
        rule_id="ChemicalFlow_v0.1.0_to_v0.2.0",
        schema_name="ChemicalFlow",
        from_version=V0_1_0,
        to_version=V0_2_0,
        description="Add optional lhv_j_per_kg, beta_factor, and schema_version",
        migrate_fn=_chemical_flow_v0_1_to_v0_2,
    ),
]:
    SCHEMA_MIGRATOR.register(_rule)

# ---------------------------------------------------------------------------
# Changelog
# ---------------------------------------------------------------------------

for _entry in [
    ChangelogEntry(
        schema_name="LedgerEntry",
        version=V0_1_0,
        change_type="ADDED",
        description="Initial LedgerEntry schema with full energy/exergy/entropy balance fields",
        breaking=False,
        affected_fields=["ledger_id", "timestamp", "boundary_id", "reference_state_id",
                         "energy_in_j", "energy_out_j", "energy_stored_delta_j",
                         "energy_rejected_j", "energy_residual_j", "exergy_in_j",
                         "useful_exergy_j", "stored_exergy_delta_j", "recovered_exergy_j",
                         "rejected_exergy_j", "destroyed_exergy_j", "exergy_residual_j",
                         "entropy_generated_j_per_k", "flags", "confidence"],
        released_at=_RELEASE_DATE_V0_1_0,
    ),
    ChangelogEntry(
        schema_name="LedgerEntry",
        version=V0_2_0,
        change_type="ADDED",
        description="Added optional carbon_footprint_kg_co2 and schema_version fields",
        breaking=False,
        affected_fields=["carbon_footprint_kg_co2", "schema_version"],
        released_at=_RELEASE_DATE_V0_2_0,
    ),
    ChangelogEntry(
        schema_name="LedgerEntry",
        version=V0_3_0,
        change_type="ADDED",
        description="Added optional cost_currency field for marginal cost tracking",
        breaking=False,
        affected_fields=["cost_currency"],
        released_at=_RELEASE_DATE_V0_3_0,
    ),
    ChangelogEntry(
        schema_name="ExergyFlow",
        version=V0_1_0,
        change_type="ADDED",
        description="Initial ExergyFlow schema",
        breaking=False,
        affected_fields=["flow_id", "carrier", "source_node_id", "target_node_id",
                         "energy_rate_w", "exergy_rate_w", "quality_factor",
                         "quality_grade", "boundary_id", "reference_state_id", "flags"],
        released_at=_RELEASE_DATE_V0_1_0,
    ),
    ChangelogEntry(
        schema_name="ExergyFlow",
        version=V0_2_0,
        change_type="ADDED",
        description="Added optional carbon_intensity_kg_co2_per_j for carbon-aware dispatch",
        breaking=False,
        affected_fields=["carbon_intensity_kg_co2_per_j", "schema_version"],
        released_at=_RELEASE_DATE_V0_2_0,
    ),
    ChangelogEntry(
        schema_name="ReferenceState",
        version=V0_1_0,
        change_type="ADDED",
        description="Initial ReferenceState schema with ambient conditions and grid context",
        breaking=False,
        affected_fields=["reference_state_id", "timestamp", "ambient_temperature_k",
                         "ambient_pressure_pa", "relative_humidity", "sky_temperature_k",
                         "nominal_grid_voltage_v", "nominal_grid_frequency_hz",
                         "marginal_carbon_kg_per_kwh", "marginal_price_per_kwh",
                         "confidence", "valid_until", "notes"],
        released_at=_RELEASE_DATE_V0_1_0,
    ),
    ChangelogEntry(
        schema_name="ReferenceState",
        version=V0_2_0,
        change_type="ADDED",
        description="Added optional soil_temperature_k for ground-source heat pump accounting",
        breaking=False,
        affected_fields=["soil_temperature_k", "schema_version"],
        released_at=_RELEASE_DATE_V0_2_0,
    ),
    ChangelogEntry(
        schema_name="Boundary",
        version=V0_1_0,
        change_type="ADDED",
        description="Initial Boundary schema",
        breaking=False,
        affected_fields=["boundary_id", "boundary_type", "included_entity_ids",
                         "excluded_entity_ids", "reference_state_id",
                         "accounting_period_start", "accounting_period_end"],
        released_at=_RELEASE_DATE_V0_1_0,
    ),
    ChangelogEntry(
        schema_name="Boundary",
        version=V0_2_0,
        change_type="ADDED",
        description="Added optional description field",
        breaking=False,
        affected_fields=["description", "schema_version"],
        released_at=_RELEASE_DATE_V0_2_0,
    ),
    ChangelogEntry(
        schema_name="ChemicalFlow",
        version=V0_1_0,
        change_type="ADDED",
        description="Initial ChemicalFlow schema with model_id and reference_environment_id",
        breaking=False,
        affected_fields=["flow_id", "mass_flow_kg_s", "specific_chemical_exergy_j_per_kg",
                         "model_id", "reference_environment_id", "boundary_id",
                         "reference_state_id"],
        released_at=_RELEASE_DATE_V0_1_0,
    ),
    ChangelogEntry(
        schema_name="ChemicalFlow",
        version=V0_2_0,
        change_type="ADDED",
        description="Added optional lhv_j_per_kg and beta_factor for fuel model transparency",
        breaking=False,
        affected_fields=["lhv_j_per_kg", "beta_factor", "schema_version"],
        released_at=_RELEASE_DATE_V0_2_0,
    ),
    ChangelogEntry(
        schema_name="BatteryState",
        version=V0_1_0,
        change_type="ADDED",
        description="Initial BatteryState schema",
        breaking=False,
        affected_fields=["storage_id", "stored_energy_j", "soc", "soh",
                         "reserve_energy_j", "boundary_id", "reference_state_id"],
        released_at=_RELEASE_DATE_V0_1_0,
    ),
    ChangelogEntry(
        schema_name="ThermalStorageState",
        version=V0_1_0,
        change_type="ADDED",
        description="Initial ThermalStorageState schema with stratified layers",
        breaking=False,
        affected_fields=["storage_id", "layers", "boundary_id", "reference_state_id"],
        released_at=_RELEASE_DATE_V0_1_0,
    ),
]:
    SCHEMA_CHANGELOG.add(_entry)
