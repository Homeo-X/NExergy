"""Tests for fuel chemical exergy via LHV and beta factors."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from eie.chemical.combustion import (
    ElementalComposition,
    GaseousMixtureComponent,
    adiabatic_flame_temperature_estimate_k,
    beta_liquid_fuel,
    beta_solid_fuel,
    fuel_chemical_exergy_j_per_kg_from_lhv,
    gaseous_mixture_beta,
    gaseous_mixture_chemical_exergy_j_per_mol,
    gaseous_mixture_lhv_j_per_mol,
    stoichiometric_air_fuel_ratio_kg_kg,
)
from eie.core.errors import DomainError


# ── ElementalComposition ─────────────────────────────────────────────────────

def test_elemental_composition_pure_carbon():
    comp = ElementalComposition(carbon=1.0)
    assert comp.h_c_ratio == pytest.approx(0.0, abs=1e-12)


def test_elemental_composition_methane_like():
    # Approx CH4 by mass: C=0.749, H=0.251
    comp = ElementalComposition(carbon=0.749, hydrogen=0.251)
    # H/C atom ratio = (0.251/1.008)/(0.749/12.011) ≈ 3.99
    assert 3.5 < comp.h_c_ratio < 4.5


def test_elemental_composition_rejects_sum_over_one():
    with pytest.raises(DomainError, match="sum to"):
        ElementalComposition(carbon=0.7, hydrogen=0.4)  # 1.1 > 1


def test_elemental_composition_rejects_negative():
    with pytest.raises(DomainError):
        ElementalComposition(carbon=-0.1)


def test_h_c_ratio_raises_for_zero_carbon():
    comp = ElementalComposition(hydrogen=0.5)
    with pytest.raises(DomainError, match="H/C ratio undefined"):
        _ = comp.h_c_ratio


# ── Beta factors ─────────────────────────────────────────────────────────────

def test_beta_liquid_for_pure_hydrocarbon_close_to_one():
    # Crude oil proxy: C~0.85, H~0.12 (no O, S, N)
    comp = ElementalComposition(carbon=0.85, hydrogen=0.12)
    beta = beta_liquid_fuel(comp)
    assert 1.02 < beta < 1.10


def test_beta_solid_for_bituminous_coal():
    # Typical bituminous coal: C=0.78, H=0.05, O=0.08, N=0.015, S=0.025
    comp = ElementalComposition(
        carbon=0.78, hydrogen=0.05, oxygen=0.08, nitrogen=0.015, sulphur=0.025
    )
    beta = beta_solid_fuel(comp)
    assert 1.05 < beta < 1.15


def test_beta_solid_for_dry_wood_biomass():
    # Dry pine: C≈0.50, H≈0.06, O≈0.43
    comp = ElementalComposition(carbon=0.50, hydrogen=0.06, oxygen=0.43)
    beta = beta_solid_fuel(comp)
    assert 1.01 < beta < 1.20


def test_fuel_chemical_exergy_from_lhv_is_proportional_to_beta():
    comp = ElementalComposition(carbon=0.85, hydrogen=0.12)
    lhv = 42_000_000.0  # J/kg typical crude oil
    beta = beta_liquid_fuel(comp)
    exergy = fuel_chemical_exergy_j_per_kg_from_lhv(lhv, beta)
    assert exergy == pytest.approx(beta * lhv, rel=1e-9)
    assert exergy > lhv  # beta > 1 for pure hydrocarbons


def test_fuel_chemical_exergy_raises_for_negative_lhv():
    with pytest.raises(DomainError):
        fuel_chemical_exergy_j_per_kg_from_lhv(-1.0, 1.05)


def test_fuel_chemical_exergy_raises_for_zero_beta():
    with pytest.raises(DomainError):
        fuel_chemical_exergy_j_per_kg_from_lhv(40_000_000.0, 0.0)


# ── Gaseous mixture ──────────────────────────────────────────────────────────

def _natural_gas_components() -> list[GaseousMixtureComponent]:
    return [
        GaseousMixtureComponent("CH4_g",   0.89),
        GaseousMixtureComponent("C2H6_g",  0.06),
        GaseousMixtureComponent("N2_g",    0.03),
        GaseousMixtureComponent("CO2_g",   0.02),
    ]


def test_natural_gas_mixture_exergy_is_positive():
    components = _natural_gas_components()
    exergy = gaseous_mixture_chemical_exergy_j_per_mol(components)
    assert exergy > 0


def test_natural_gas_mixture_lhv_is_positive():
    components = _natural_gas_components()
    lhv = gaseous_mixture_lhv_j_per_mol(components)
    assert lhv > 0


def test_natural_gas_beta_slightly_above_one():
    components = _natural_gas_components()
    beta = gaseous_mixture_beta(components)
    # Natural gas: beta ≈ 1.04
    assert 1.00 < beta < 1.10


def test_gaseous_mixture_fractions_must_sum_to_one():
    components = [
        GaseousMixtureComponent("CH4_g", 0.5),
        GaseousMixtureComponent("H2_g",  0.3),
        # sum = 0.8, not 1.0
    ]
    with pytest.raises(DomainError, match="sum to"):
        gaseous_mixture_chemical_exergy_j_per_mol(components)


def test_pure_hydrogen_mixture_beta_matches_tabulated():
    components = [GaseousMixtureComponent("H2_g", 1.0)]
    exergy = gaseous_mixture_chemical_exergy_j_per_mol(components, include_mixing_exergy=False)
    from eie.chemical.pure_substances import standard_chemical_exergy_j_per_mol
    assert exergy == pytest.approx(standard_chemical_exergy_j_per_mol("H2", "g"), rel=1e-9)


def test_inert_gas_mixture_lhv_is_zero():
    components = [
        GaseousMixtureComponent("N2_g",  0.78),
        GaseousMixtureComponent("O2_g",  0.21),
        GaseousMixtureComponent("CO2_g", 0.01),
    ]
    lhv = gaseous_mixture_lhv_j_per_mol(components)
    assert lhv == pytest.approx(0.0, abs=1.0)


# ── Stoichiometric AFR ───────────────────────────────────────────────────────

def test_stoichiometric_afr_for_pure_carbon():
    # C + O2 → CO2: O2_req = (32/12) = 2.667 kg_O2/kg_C, AFR = 2.667/0.232 ≈ 11.5
    comp = ElementalComposition(carbon=1.0)
    afr = stoichiometric_air_fuel_ratio_kg_kg(comp)
    assert 11.0 < afr < 12.0


def test_stoichiometric_afr_for_methane_like():
    # CH4 mass fractions: C=0.749, H=0.251
    comp = ElementalComposition(carbon=0.749, hydrogen=0.251)
    afr = stoichiometric_air_fuel_ratio_kg_kg(comp)
    # Methane AFR ≈ 17.2
    assert 15.0 < afr < 19.0


def test_adiabatic_flame_temperature_is_above_reference():
    comp = ElementalComposition(carbon=0.85, hydrogen=0.12)
    t_ad = adiabatic_flame_temperature_estimate_k(comp, lhv_j_per_kg=42_000_000.0)
    assert t_ad > 2000.0  # should be well above ambient


def test_adiabatic_flame_temperature_decreases_with_excess_air():
    comp = ElementalComposition(carbon=0.85, hydrogen=0.12)
    lhv = 42_000_000.0
    t_stoich = adiabatic_flame_temperature_estimate_k(comp, lhv_j_per_kg=lhv)
    t_excess = adiabatic_flame_temperature_estimate_k(
        comp, lhv_j_per_kg=lhv, excess_air_fraction=0.5
    )
    assert t_excess < t_stoich


# ── Property-based tests ─────────────────────────────────────────────────────

@settings(deadline=None, max_examples=100)
@given(
    c=st.floats(min_value=0.1, max_value=0.9),
    h=st.floats(min_value=0.01, max_value=0.15),
)
def test_beta_liquid_is_positive_for_valid_inputs(c, h):
    if c + h > 0.99:
        return
    comp = ElementalComposition(carbon=c, hydrogen=h)
    beta = beta_liquid_fuel(comp)
    assert beta > 0


@settings(deadline=None, max_examples=100)
@given(
    lhv=st.floats(min_value=1e6, max_value=1e8),
    beta=st.floats(min_value=0.9, max_value=1.5),
)
def test_fuel_exergy_from_lhv_is_proportional(lhv, beta):
    exergy = fuel_chemical_exergy_j_per_kg_from_lhv(lhv, beta)
    assert exergy == pytest.approx(beta * lhv, rel=1e-9)
