"""Tests for the semantic version system."""

from __future__ import annotations

import pytest

from eie.core.errors import DomainError
from eie.schema.version import V0_1_0, V0_2_0, V0_3_0, V1_0_0, SemanticVersion


def test_version_ordering():
    assert V0_1_0 < V0_2_0 < V0_3_0 < V1_0_0
    assert V0_1_0 <= V0_1_0
    assert V0_2_0 > V0_1_0


def test_version_string_format():
    assert str(V0_1_0) == "0.1.0"
    assert str(V1_0_0) == "1.0.0"


def test_version_parse():
    v = SemanticVersion.parse("1.2.3")
    assert v.major == 1
    assert v.minor == 2
    assert v.patch == 3


def test_version_parse_rejects_invalid():
    with pytest.raises(ValueError):
        SemanticVersion.parse("1.2")
    with pytest.raises(ValueError):
        SemanticVersion.parse("a.b.c")


def test_version_rejects_negative():
    with pytest.raises(ValueError):
        SemanticVersion(major=-1, minor=0, patch=0)


def test_is_compatible_with_same_major():
    # v0.2.0 reader can read v0.1.0 data (forward-compatible reader)
    assert V0_2_0.is_compatible_with(V0_1_0)
    # But v0.1.0 reader cannot read v0.2.0 data
    assert not V0_1_0.is_compatible_with(V0_2_0)


def test_is_compatible_with_same_version():
    assert V0_1_0.is_compatible_with(V0_1_0)


def test_is_compatible_with_different_major():
    assert not V1_0_0.is_compatible_with(V0_1_0)
    assert not V0_1_0.is_compatible_with(V1_0_0)


def test_is_breaking_change_from():
    assert V1_0_0.is_breaking_change_from(V0_1_0)
    assert not V0_2_0.is_breaking_change_from(V0_1_0)
    assert not V0_1_0.is_breaking_change_from(V0_1_0)


def test_bump_major():
    v = SemanticVersion(0, 5, 3).bump_major()
    assert v == SemanticVersion(1, 0, 0)


def test_bump_minor():
    v = SemanticVersion(0, 1, 0).bump_minor()
    assert v == SemanticVersion(0, 2, 0)


def test_bump_patch():
    v = SemanticVersion(0, 1, 0).bump_patch()
    assert v == SemanticVersion(0, 1, 1)


def test_version_equality_and_hashing():
    a = SemanticVersion(0, 1, 0)
    b = SemanticVersion(0, 1, 0)
    assert a == b
    # Should be usable as dict key
    d = {a: "value"}
    assert d[b] == "value"
