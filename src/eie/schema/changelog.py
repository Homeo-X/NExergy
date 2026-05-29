"""Structured schema changelog.

The changelog records every change to every schema version in the EIE
library.  It is the authoritative source for:
- What changed and when (for code reviewers and API consumers)
- Which changes are breaking (for semantic version bumping)
- Which fields are affected (for migration rule authoring)

Changelog entries are immutable and append-only (no corrections,
no deletions).  This mirrors the AuditLedger philosophy.

Usage
─────
    changelog = SchemaChangelog()
    changelog.add(ChangelogEntry(
        schema_name="LedgerEntry",
        version=V0_2_0,
        change_type="ADDED",
        description="Added optional schema_version field",
        breaking=False,
        affected_fields=["schema_version"],
        released_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    ))
    changes_since_v0_1_0 = changelog.since("LedgerEntry", V0_1_0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from eie.core.errors import DomainError
from eie.schema.version import ChangeType, SemanticVersion


@dataclass(frozen=True)
class ChangelogEntry:
    """One structured entry in the schema changelog."""

    schema_name: str
    version: SemanticVersion
    change_type: ChangeType
    description: str
    breaking: bool
    affected_fields: list[str]
    released_at: datetime
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.schema_name:
            raise DomainError("ChangelogEntry.schema_name is required")
        if not self.description:
            raise DomainError("ChangelogEntry.description is required")
        if self.change_type not in ("ADDED", "CHANGED", "DEPRECATED", "REMOVED", "FIXED", "SECURITY"):
            raise DomainError(
                f"change_type must be one of ADDED/CHANGED/DEPRECATED/REMOVED/FIXED/SECURITY, "
                f"got {self.change_type!r}"
            )
        if not isinstance(self.released_at, datetime):
            raise DomainError("released_at must be a datetime")
        if self.breaking and self.change_type not in ("REMOVED", "CHANGED"):
            raise DomainError(
                "only REMOVED or CHANGED entries can be marked as breaking"
            )


class SchemaChangelog:
    """Append-only in-memory changelog.

    Entries are stored in insertion order.  No entry can be removed or
    modified after insertion (same principle as AuditLedger).
    """

    def __init__(self) -> None:
        self._entries: list[ChangelogEntry] = []

    def add(self, entry: ChangelogEntry) -> None:
        self._entries.append(entry)

    def all_entries(self) -> list[ChangelogEntry]:
        return list(self._entries)

    def for_schema(self, schema_name: str) -> list[ChangelogEntry]:
        return [e for e in self._entries if e.schema_name == schema_name]

    def since(
        self,
        schema_name: str,
        from_version: SemanticVersion,
        *,
        inclusive: bool = False,
    ) -> list[ChangelogEntry]:
        """Return entries for `schema_name` with version > from_version.

        If inclusive=True, return entries with version >= from_version.
        """
        entries = self.for_schema(schema_name)
        if inclusive:
            return [e for e in entries if e.version >= from_version]
        return [e for e in entries if e.version > from_version]

    def breaking_changes_between(
        self,
        schema_name: str,
        from_version: SemanticVersion,
        to_version: SemanticVersion,
    ) -> list[ChangelogEntry]:
        """Return all breaking changes between two versions (exclusive start, inclusive end)."""
        return [
            e
            for e in self.for_schema(schema_name)
            if e.breaking and from_version < e.version <= to_version
        ]

    def has_breaking_changes(
        self,
        schema_name: str,
        from_version: SemanticVersion,
        to_version: SemanticVersion,
    ) -> bool:
        return bool(self.breaking_changes_between(schema_name, from_version, to_version))

    def entries_for_version(
        self, schema_name: str, version: SemanticVersion
    ) -> list[ChangelogEntry]:
        return [
            e for e in self.for_schema(schema_name) if e.version == version
        ]

    def all_schema_names(self) -> list[str]:
        return sorted({e.schema_name for e in self._entries})

    def summary(self) -> dict[str, list[str]]:
        """Return schema_name → list of 'v{version}: {change_type} {description}' strings."""
        result: dict[str, list[str]] = {}
        for e in self._entries:
            result.setdefault(e.schema_name, []).append(
                f"v{e.version}: [{e.change_type}] {e.description}"
                + (" [BREAKING]" if e.breaking else "")
            )
        return result
