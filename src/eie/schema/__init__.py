"""Schema versioning for the Exergy Intelligence Engine.

Provides semantic version tracking, field-level declarations, a schema
registry, forward-migration chains, and a structured changelog for all
EIE domain objects.

Public API
──────────
Versioning:
    SemanticVersion, V0_1_0, V0_2_0, V0_3_0, V1_0_0

Declaration:
    FieldDeclaration, SchemaDeclaration

Registry:
    SchemaRegistry

Migration:
    MigrationRule, MigrationEvent, VersionedRecord, SchemaMigrator

Changelog:
    ChangelogEntry, SchemaChangelog

Domain singletons:
    SCHEMA_REGISTRY, SCHEMA_MIGRATOR, SCHEMA_CHANGELOG
"""

from eie.schema.changelog import ChangelogEntry, SchemaChangelog
from eie.schema.declaration import FieldDeclaration, SchemaDeclaration
from eie.schema.domain_schemas import SCHEMA_CHANGELOG, SCHEMA_MIGRATOR, SCHEMA_REGISTRY
from eie.schema.migration import MigrationEvent, MigrationRule, SchemaMigrator, VersionedRecord
from eie.schema.registry import SchemaRegistry
from eie.schema.version import (
    V0_1_0,
    V0_2_0,
    V0_3_0,
    V1_0_0,
    ChangeType,
    SemanticVersion,
)

__all__ = [
    # versions
    "V0_1_0",
    "V0_2_0",
    "V0_3_0",
    "V1_0_0",
    "ChangeType",
    "SemanticVersion",
    # declaration
    "FieldDeclaration",
    "SchemaDeclaration",
    # registry
    "SchemaRegistry",
    # migration
    "MigrationEvent",
    "MigrationRule",
    "SchemaMigrator",
    "VersionedRecord",
    # changelog
    "ChangelogEntry",
    "SchemaChangelog",
    # domain singletons
    "SCHEMA_CHANGELOG",
    "SCHEMA_MIGRATOR",
    "SCHEMA_REGISTRY",
]
