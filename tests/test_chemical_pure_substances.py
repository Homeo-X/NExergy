"""Tests for the pure-substance chemical exergy database."""

from __future__ import annotations

import pytest

from eie.chemical.pure_substances import (
    PURE_SUBSTANCE_DB,
    get_by_formula,
    get_by_id,
    list_substances,
    search_by_formula_prefix,
    standard_chemical_exergy_j_per_kg,
    standard_chemical_exergy_j_per_mol,
)
from eie.core.errors import DomainError


# ── Basic lookups ────────────────────────────────────────────────────────────

def test_hydrogen_gas_exergy_is_in_szargut_range():
    record = get_by_formula("H2", "g")
    # Szargut 2005: 236.09 kJ/mol
    assert record.exergy_kj_per_mol == pytest.approx(236.09, rel=1e-3)


def test_methane_gas_exergy_kj_per_mol():
    record = get_by_formula("CH4", "g")
    # Szargut 2005: 831.65 kJ/mol
    assert record.exergy_kj_per_mol == pytest.approx(831.65, rel=1e-3)


def test_co2_gas_exergy_kj_per_mol():
    record = get_by_formula("CO2", "g")
    assert record.exergy_kj_per_mol == pytest.approx(19.48, rel=1e-3)


def test_water_liquid_is_reference_substance():
    record = get_by_formula("H2O", "l")
    assert record.is_reference_substance
    assert record.exergy_kj_per_mol == pytest.approx(0.90, rel=1e-3)


def test_specific_exergy_j_per_kg_for_hydrogen():
    x = standard_chemical_exergy_j_per_kg("H2", "g")
    # 236090 J/mol / (0.002016 kg/mol) ≈ 117.1 MJ/kg
    assert 110_000_000 < x < 125_000_000


def test_specific_exergy_j_per_mol_for_methane():
    x = standard_chemical_exergy_j_per_mol("CH4", "g")
    assert x == pytest.approx(831_650.0, rel=1e-3)


def test_get_by_id():
    record = get_by_id("CH4_g")
    assert record.formula == "CH4"
    assert record.phase == "g"


def test_get_by_id_unknown_raises():
    with pytest.raises(DomainError, match="no chemical exergy data"):
        get_by_id("XYZFAKE_g")


def test_get_by_formula_unknown_raises():
    with pytest.raises(DomainError, match="no chemical exergy data"):
        get_by_formula("XYZ", "g")


def test_get_by_formula_wrong_phase_raises():
    with pytest.raises(DomainError):
        get_by_formula("CH4", "l")  # methane is only registered as gas


def test_list_substances_returns_sorted_list():
    substances = list_substances()
    assert len(substances) > 20
    assert substances == sorted(substances)
    assert "CH4_g" in substances
    assert "H2_g" in substances


def test_search_by_formula_prefix():
    results = search_by_formula_prefix("C2")
    formulas = [r.formula for r in results]
    assert all(f.startswith("C2") for f in formulas)
    assert len(results) >= 3  # C2H2, C2H4, C2H6 at minimum


def test_all_database_exergies_are_non_negative():
    for substance_id, record in PURE_SUBSTANCE_DB.items():
        assert record.exergy_j_per_mol >= 0, f"{substance_id} has negative exergy"


def test_all_database_molar_masses_are_positive():
    for substance_id, record in PURE_SUBSTANCE_DB.items():
        assert record.molar_mass_g_mol > 0, f"{substance_id} has non-positive molar mass"


def test_exergy_j_per_kg_derived_from_j_per_mol():
    record = get_by_formula("CH4", "g")
    expected = record.exergy_j_per_mol / (record.molar_mass_g_mol * 1.0e-3)
    assert record.exergy_j_per_kg == pytest.approx(expected, rel=1e-9)


def test_fuel_exergies_are_much_larger_than_inert_gas_exergies():
    ch4_exergy = get_by_formula("CH4", "g").exergy_j_per_mol
    o2_exergy = get_by_formula("O2", "g").exergy_j_per_mol
    n2_exergy = get_by_formula("N2", "g").exergy_j_per_mol
    assert ch4_exergy > 100 * o2_exergy
    assert ch4_exergy > 100 * n2_exergy


def test_ethanol_liquid_exergy_kj_per_mol():
    record = get_by_formula("C2H5OH", "l")
    # Szargut 2005: 1363 kJ/mol
    assert record.exergy_kj_per_mol == pytest.approx(1363.0, rel=1e-3)


def test_carbon_solid_exergy():
    record = get_by_id("C_s")
    # Szargut 2005: 410.26 kJ/mol
    assert record.exergy_kj_per_mol == pytest.approx(410.26, rel=1e-3)
