"""Schema registry — central catalogue of all schema versions.

The registry maps (schema_name, version) pairs to SchemaDeclarations.
It enforces monotonic version ordering and detects version conflicts.

Usage
─────
    registry = SchemaRegistry()
    registry.register(ledger_entry_v0_1_0)
    registry.register(ledger_entry_v0_2_0)

    current = registry.current_version("LedgerEntry")
    history = registry.all_versions("LedgerEntry")
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eie.core.errors import DomainError
from eie.schema.declaration import SchemaDeclaration
from eie.schema.version import SemanticVersion


@dataclass
class SchemaRegistry:
    """In-memory registry of all known schema versions.

    Thread-safety: not thread-safe; intended for single-threaded usage
    or module-level initialisation.
    """

    _declarations: dict[str, dict[SemanticVersion, SchemaDeclaration]] = field(
        default_factory=dict
    )

    def register(self, declaration: SchemaDeclaration) -> None:
        """Register a schema declaration.

        Raises DomainError if the same (name, version) is already registered.
        """
        name = declaration.schema_name
        version = declaration.schema_version
        if name not in self._declarations:
            self._declarations[name] = {}
        if version in self._declarations[name]:
            raise DomainError(
                f"schema {name!r} version {version} is already registered"
            )
        self._declarations[name][version] = declaration

    def get(self, schema_name: str, version: SemanticVersion) -> SchemaDeclaration:
        """Return the schema declaration for an exact (name, version) pair."""
        try:
            return self._declarations[schema_name][version]
        except KeyError as exc:
            raise DomainError(
                f"no schema {schema_name!r} at version {version} registered"
            ) from exc

    def current_version(self, schema_name: str) -> SemanticVersion:
        """Return the highest registered version for this schema."""
        if schema_name not in self._declarations:
            raise DomainError(f"no schema {schema_name!r} registered")
        return max(self._declarations[schema_name])

    def current(self, schema_name: str) -> SchemaDeclaration:
        """Return the schema declaration for the current (highest) version."""
        return self.get(schema_name, self.current_version(schema_name))

    def all_versions(self, schema_name: str) -> list[SchemaDeclaration]:
        """Return all versions for this schema, sorted ascending."""
        if schema_name not in self._declarations:
            raise DomainError(f"no schema {schema_name!r} registered")
        return [
            self._declarations[schema_name][v]
            for v in sorted(self._declarations[schema_name])
        ]

    def known_schemas(self) -> list[str]:
        """Return sorted list of all registered schema names."""
        return sorted(self._declarations)

    def is_registered(self, schema_name: str, version: SemanticVersion | None = None) -> bool:
        """Return True if the schema (and optionally a specific version) is registered."""
        if schema_name not in self._declarations:
            return False
        if version is None:
            return True
        return version in self._declarations[schema_name]

    def is_field_present_in(
        self, schema_name: str, field_name: str, version: SemanticVersion
    ) -> bool:
        """Return True when `field_name` is present in the schema at `version`."""
        decl = self.get(schema_name, version)
        if not decl.has_field(field_name):
            return False
        return decl.field(field_name).is_present_in(version)

    def compatible_version(
        self, schema_name: str, reader_version: SemanticVersion
    ) -> list[SemanticVersion]:
        """Return all writer versions that `reader_version` can read."""
        all_v = [v for v in self._declarations.get(schema_name, {})]
        return [v for v in all_v if reader_version.is_compatible_with(v)]

    def summary(self) -> dict[str, list[str]]:
        """Return a summary dict: schema_name → list of version strings."""
        return {
            name: [str(v) for v in sorted(versions)]
            for name, versions in self._declarations.items()
        }
