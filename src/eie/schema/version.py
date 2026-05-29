"""Semantic versioning for EIE domain schemas.

Follows SemVer 2.0.0 (https://semver.org):
  MAJOR — incompatible API change (breaks existing records without migration)
  MINOR — backward-compatible addition
  PATCH — backward-compatible fix

Compatibility rule implemented here:
  Version A is compatible with (can read) version B if and only if
  A.major == B.major and A >= B.

This means a v0.2.0 reader CAN read v0.1.0 data (forward-compatible
reader), but a v0.1.0 reader CANNOT read v0.2.0 data (it does not
understand new optional fields).  Major-version bumps always break
compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, order=True)
class SemanticVersion:
    """Immutable semantic version with comparison operators.

    Ordering follows (major, minor, patch) tuple ordering.
    """

    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        for name in ("major", "minor", "patch"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"SemanticVersion.{name} must be a non-negative integer")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def parse(cls, s: str) -> SemanticVersion:
        """Parse 'MAJOR.MINOR.PATCH' string."""
        parts = s.strip().split(".")
        if len(parts) != 3:
            raise ValueError(f"cannot parse version string {s!r}; expected MAJOR.MINOR.PATCH")
        try:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError as exc:
            raise ValueError(f"version string {s!r} contains non-integer part") from exc
        return cls(major, minor, patch)

    def is_compatible_with(self, other: SemanticVersion) -> bool:
        """Return True if self can read records written by `other`.

        A reader is compatible with a writer if they share the same major
        version and the reader version >= writer version.
        """
        return self.major == other.major and self >= other

    def is_breaking_change_from(self, other: SemanticVersion) -> bool:
        """Return True when upgrading from `other` to self is a breaking change."""
        return self.major > other.major

    def bump_major(self) -> SemanticVersion:
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> SemanticVersion:
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> SemanticVersion:
        return SemanticVersion(self.major, self.minor, self.patch + 1)


ChangeType = Literal["ADDED", "CHANGED", "DEPRECATED", "REMOVED", "FIXED", "SECURITY"]


# Common version constants used throughout the library.
V0_1_0 = SemanticVersion(0, 1, 0)
V0_2_0 = SemanticVersion(0, 2, 0)
V0_3_0 = SemanticVersion(0, 3, 0)
V1_0_0 = SemanticVersion(1, 0, 0)
