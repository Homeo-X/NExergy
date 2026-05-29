"""Tests for the chemical reference environment module."""

from __future__ import annotations

import pytest

from eie.chemical.reference_environment import (
    MOLAR_MASS_G_MOL,
    REFERENCE_SUBSTANCES,
    SZARGUT_RE,
    R_J_MOL_K,
    T0_K,
    P0_PA,
    ReferenceEnvironment,
    ReferenceSubstance,
)
from eie.core.errors import DomainError


def test_standard_constants_are_physically_reasonable():
    assert 8.31 < R_J_MOL_K < 8.32
    assert T0_K == pytest.approx(298.15)
    assert P0_PA == pytest.approx(101_325.0)


def test_szargut_re_has_correct_temperature_and_pressure():
    assert SZARGUT_RE.temperature_k == T0_K
    assert SZARGUT_RE.pressure_pa == P0_PA
    assert "Szargut" in SZARGUT_RE.name


def test_key_reference_substances_present():
    for element in ("C", "H", "O", "N", "S", "Ar"):
        rs = SZARGUT_RE.reference_substance(element)
        assert rs.element == element
        assert rs.formula
        assert rs.molar_mass_g_mol > 0


def test_missing_reference_substance_raises():
    with pytest.raises(DomainError, match="no reference substance"):
        SZARGUT_RE.reference_substance("Xy")


def test_atmospheric_partial_pressure_for_oxygen():
    p_o2 = SZARGUT_RE.atmospheric_partial_pressure_pa("O")
    # O2 is ~20.9% of atmosphere
    assert 20_000 < p_o2 < 22_000


def test_atmospheric_partial_pressure_for_nitrogen():
    p_n2 = SZARGUT_RE.atmospheric_partial_pressure_pa("N")
    # N2 is ~78% of atmosphere
    assert 78_000 < p_n2 < 80_000


def test_atmospheric_partial_pressure_for_co2():
    p_co2 = SZARGUT_RE.atmospheric_partial_pressure_pa("C")
    # CO2 is ~420 ppm
    assert 40 < p_co2 < 50


def test_restricted_dead_state_exergy_for_oxygen():
    x_o2 = SZARGUT_RE.restricted_dead_state_exergy_j_mol("O")
    # Should be ~3970 J/mol (Szargut value for O2)
    assert 3_500 < x_o2 < 4_500


def test_restricted_dead_state_exergy_for_nitrogen():
    x_n2 = SZARGUT_RE.restricted_dead_state_exergy_j_mol("N")
    # Szargut N2: ~720 J/mol
    assert 600 < x_n2 < 900


def test_solid_reference_substance_raises_on_partial_pressure():
    with pytest.raises(DomainError, match="not a gas"):
        SZARGUT_RE.atmospheric_partial_pressure_pa("S")


def test_solid_reference_substance_raises_on_restricted_dead_state():
    with pytest.raises(DomainError, match="requires a gas"):
        SZARGUT_RE.restricted_dead_state_exergy_j_mol("S")


def test_molar_mass_registry_covers_common_molecules():
    for formula in ("H2", "O2", "N2", "CO2", "CH4", "H2O", "C2H6", "NH3"):
        assert formula in MOLAR_MASS_G_MOL
        assert MOLAR_MASS_G_MOL[formula] > 0


def test_molar_mass_missing_raises():
    with pytest.raises(DomainError, match="no molar mass registered"):
        SZARGUT_RE.molar_mass_g_mol("XyZfake")


def test_reference_environment_validation():
    with pytest.raises(DomainError):
        ReferenceEnvironment(name="", temperature_k=298.15, pressure_pa=101325.0)
    with pytest.raises(DomainError):
        ReferenceEnvironment(name="bad-T", temperature_k=-10.0, pressure_pa=101325.0)
    with pytest.raises(DomainError):
        ReferenceEnvironment(name="bad-P", temperature_k=298.15, pressure_pa=0.0)


def test_reference_substance_validation():
    with pytest.raises(DomainError):
        ReferenceSubstance(
            element="", formula="O2", phase="g", description="bad",
            molar_mass_g_mol=32.0, concentration_in_re=0.2
        )
    with pytest.raises(DomainError):
        ReferenceSubstance(
            element="O", formula="O2", phase="g", description="bad",
            molar_mass_g_mol=-1.0, concentration_in_re=0.2
        )
