"""Schema field and schema-level declarations.

A SchemaDeclaration describes the structure of one versioned domain object.
It is NOT a runtime enforcement mechanism — Python's type system + mypy do
that.  The declaration is a structured record used by:
  • The schema registry (to track all versions)
  • The changelog (to link entries to field changes)
  • The migration system (to know which fields exist at each version)
  • Runtime schema validation helpers

Field declarations track the lifecycle of each field:
  added_in:      first version where the field appears
  deprecated_in: version where the field is marked for removal
  removed_in:    version where the field is removed (breaks backward compat → major bump)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eie.core.errors import DomainError
from eie.schema.version import SemanticVersion


@dataclass(frozen=True)
class FieldDeclaration:
    """Declaration of one field in a schema version."""

    name: str
    type_hint: str          # Python type hint as a string (for documentation)
    required: bool          # False = has a default; may be absent in older records
    added_in: SemanticVersion
    default: Any = None     # Default value used when field is absent in older data
    deprecated_in: SemanticVersion | None = None
    removed_in: SemanticVersion | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise DomainError("FieldDeclaration.name is required")
        if not self.type_hint:
            raise DomainError("FieldDeclaration.type_hint is required")
        if self.deprecated_in is not None and self.removed_in is not None:
            if self.removed_in < self.deprecated_in:
                raise DomainError(
                    f"field {self.name!r}: removed_in {self.removed_in} is before "
                    f"deprecated_in {self.deprecated_in}"
                )

    def is_present_in(self, version: SemanticVersion) -> bool:
        """Return True when this field exists in the given schema version."""
        if version < self.added_in:
            return False
        if self.removed_in is not None and version >= self.removed_in:
            return False
        return True

    def is_deprecated_in(self, version: SemanticVersion) -> bool:
        return (
            self.deprecated_in is not None
            and version >= self.deprecated_in
            and (self.removed_in is None or version < self.removed_in)
        )


@dataclass(frozen=True)
class SchemaDeclaration:
    """Complete versioned schema declaration for one domain object type.

    Parameters
    ----------
    schema_name : str
        Unique name (e.g. "LedgerEntry", "ExergyFlow").
    schema_version : SemanticVersion
        The version this declaration describes.
    description : str
        Human-readable description of the schema purpose.
    fields : list[FieldDeclaration]
        All fields (required and optional) present in this version.
    notes : str | None
        Additional documentation.
    """

    schema_name: str
    schema_version: SemanticVersion
    description: str
    fields: list[FieldDeclaration] = field(default_factory=list)
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.schema_name:
            raise DomainError("SchemaDeclaration.schema_name is required")
        if not self.description:
            raise DomainError("SchemaDeclaration.description is required")
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise DomainError(
                f"schema {self.schema_name!r}: duplicate field names detected"
            )

    def required_fields(self) -> list[FieldDeclaration]:
        return [f for f in self.fields if f.required and f.is_present_in(self.schema_version)]

    def optional_fields(self) -> list[FieldDeclaration]:
        return [f for f in self.fields if not f.required and f.is_present_in(self.schema_version)]

    def deprecated_fields(self) -> list[FieldDeclaration]:
        return [f for f in self.fields if f.is_deprecated_in(self.schema_version)]

    def field(self, name: str) -> FieldDeclaration:
        for f in self.fields:
            if f.name == name:
                return f
        raise DomainError(
            f"schema {self.schema_name!r} v{self.schema_version} has no field {name!r}"
        )

    def has_field(self, name: str) -> bool:
        return any(f.name == name for f in self.fields)

    def validate_record(self, record: dict[str, Any]) -> list[str]:
        """Return a list of validation errors for a raw dict record.

        Does not raise; callers decide how to handle the errors.
        """
        errors: list[str] = []
        for fd in self.required_fields():
            if fd.name not in record:
                errors.append(f"required field {fd.name!r} is missing")
        for key in record:
            if not self.has_field(key):
                errors.append(f"unexpected field {key!r} not in schema v{self.schema_version}")
        return errors
