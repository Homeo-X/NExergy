"""Schema migration framework.

A MigrationRule describes how to transform a record from one schema version
to the next.  Migration is always forward (older → newer); downgrade is not
supported in v0 because older readers cannot understand new fields.

The SchemaMigrator:
1. Finds the migration path (chain of rules) from source to target version.
2. Applies each rule in sequence to the raw dict payload.
3. Returns a new VersionedRecord with the updated payload and migration history.

Rules must be pure functions: given a dict, return a new dict.
They must NOT have side effects (no I/O, no global state mutation).

Migration invariants enforced:
- No migration chain can skip a version (must be contiguous step-by-step).
- Each migration rule is applied exactly once in the chain.
- The migrated payload includes a `_migration_applied` annotation.

Immutability principle: corrections to ledger records are new entries, not
silent overwrites.  Migration of live ledger data therefore produces new
VersionedRecord objects; the original is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from eie.core.errors import DomainError
from eie.schema.version import SemanticVersion


PayloadDict = dict[str, object]
MigrationFn = Callable[[PayloadDict], PayloadDict]


@dataclass(frozen=True)
class MigrationRule:
    """One step in a migration chain.

    Parameters
    ----------
    rule_id : str
        Unique human-readable identifier (e.g. "LedgerEntry_v0.1.0_to_v0.2.0").
    schema_name : str
        The schema this rule applies to.
    from_version : SemanticVersion
        The source schema version.
    to_version : SemanticVersion
        The target schema version (must be exactly one step newer).
    description : str
        Human-readable description of what this migration does.
    migrate_fn : MigrationFn
        Pure function that transforms the raw payload dict.
        Must return a NEW dict (immutable in spirit).
    """

    rule_id: str
    schema_name: str
    from_version: SemanticVersion
    to_version: SemanticVersion
    description: str
    migrate_fn: MigrationFn

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise DomainError("MigrationRule.rule_id is required")
        if not self.schema_name:
            raise DomainError("MigrationRule.schema_name is required")
        if self.from_version >= self.to_version:
            raise DomainError(
                f"MigrationRule {self.rule_id!r}: from_version {self.from_version} "
                f"must be strictly less than to_version {self.to_version}"
            )

    def apply(self, payload: PayloadDict) -> PayloadDict:
        """Apply the migration function and annotate the result."""
        result = self.migrate_fn(dict(payload))
        result["_migration_applied"] = self.rule_id
        return result


@dataclass(frozen=True)
class MigrationEvent:
    """Record of one migration step applied to a VersionedRecord."""

    rule_id: str
    from_version: str
    to_version: str
    applied_at: datetime
    notes: str | None = None


@dataclass(frozen=True)
class VersionedRecord:
    """A domain object payload wrapped with schema version metadata.

    The record is immutable.  Migration produces a new VersionedRecord.
    The original is always preserved — corrections are new entries, not
    silent overwrites (immutability principle).

    Parameters
    ----------
    record_id : str
        Unique identifier for this logical record.
    schema_name : str
        Name of the schema this payload conforms to.
    schema_version : SemanticVersion
        The version of the schema this payload was written in.
    payload : PayloadDict
        The raw domain data (keys = field names, values = Python objects).
    migration_history : list[MigrationEvent]
        Ordered list of all migrations applied to reach this version.
    created_at : datetime
        When the record was first created (UTC).
    """

    record_id: str
    schema_name: str
    schema_version: SemanticVersion
    payload: PayloadDict
    migration_history: list[MigrationEvent] = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if not self.record_id:
            raise DomainError("VersionedRecord.record_id is required")
        if not self.schema_name:
            raise DomainError("VersionedRecord.schema_name is required")


class SchemaMigrator:
    """Applies migration chains to VersionedRecord payloads.

    Migration rules are registered step by step.  The migrator finds the
    shortest path from source to target version by following the chain.
    """

    def __init__(self) -> None:
        self._rules: dict[str, list[MigrationRule]] = {}

    def register(self, rule: MigrationRule) -> None:
        """Register a migration rule.

        Raises DomainError if a rule for the same (schema, from_version) already exists.
        """
        key = f"{rule.schema_name}:{rule.from_version}"
        if key in self._rules and any(
            r.to_version == rule.to_version for r in self._rules.get(key, [])
        ):
            raise DomainError(
                f"migration rule from {rule.from_version} to {rule.to_version} "
                f"for schema {rule.schema_name!r} is already registered"
            )
        if key not in self._rules:
            self._rules[key] = []
        self._rules[key].append(rule)

    def migration_path(
        self,
        schema_name: str,
        from_version: SemanticVersion,
        to_version: SemanticVersion,
    ) -> list[MigrationRule]:
        """Return the ordered list of rules to migrate from source to target.

        Uses BFS to find the shortest path (in practice chains are linear).
        Raises DomainError if no path exists.
        """
        if from_version == to_version:
            return []
        if from_version > to_version:
            raise DomainError(
                f"cannot migrate {schema_name!r} backwards from {from_version} to {to_version}; "
                "downgrade migrations are not supported"
            )

        # BFS over version graph
        from collections import deque
        queue: deque[tuple[SemanticVersion, list[MigrationRule]]] = deque()
        queue.append((from_version, []))
        visited: set[SemanticVersion] = {from_version}

        while queue:
            current_version, path = queue.popleft()
            key = f"{schema_name}:{current_version}"
            for rule in self._rules.get(key, []):
                if rule.to_version in visited:
                    continue
                new_path = path + [rule]
                if rule.to_version == to_version:
                    return new_path
                visited.add(rule.to_version)
                queue.append((rule.to_version, new_path))

        raise DomainError(
            f"no migration path for schema {schema_name!r} from {from_version} to {to_version}; "
            f"registered rules: {[k for k in self._rules if k.startswith(schema_name)]}"
        )

    def migrate(
        self,
        record: VersionedRecord,
        to_version: SemanticVersion,
        *,
        notes: str | None = None,
    ) -> VersionedRecord:
        """Migrate a VersionedRecord to the target version.

        Returns a NEW VersionedRecord with the migrated payload and updated
        migration history.  The original record is unchanged.
        """
        path = self.migration_path(record.schema_name, record.schema_version, to_version)
        if not path:
            return record   # already at target version

        payload = dict(record.payload)
        history = list(record.migration_history)
        now = datetime.now(timezone.utc)

        for rule in path:
            payload = rule.apply(payload)
            history.append(
                MigrationEvent(
                    rule_id=rule.rule_id,
                    from_version=str(rule.from_version),
                    to_version=str(rule.to_version),
                    applied_at=now,
                    notes=notes,
                )
            )

        return VersionedRecord(
            record_id=record.record_id,
            schema_name=record.schema_name,
            schema_version=to_version,
            payload=payload,
            migration_history=history,
            created_at=record.created_at,
        )

    def can_migrate(
        self, schema_name: str, from_version: SemanticVersion, to_version: SemanticVersion
    ) -> bool:
        """Return True if a migration path exists."""
        try:
            self.migration_path(schema_name, from_version, to_version)
            return True
        except DomainError:
            return False
