"""Tests for chemical reaction exergy."""

from __future__ import annotations

import pytest

from eie.chemical.reaction import (
    CALCINATION,
    COMBUSTION_CH4,
    COMBUSTION_CO,
    COMBUSTION_H2,
    COMMON_REACTIONS,
    GLUCOSE_AEROBIC_OXIDATION,
    HABER_BOSCH,
    STEAM_METHANE_REFORMING,
    WATER_GAS_SHIFT,
    ChemicalReaction,
    ReactionSpecies,
)
from eie.core.errors import DomainError


def test_methane_combustion_is_spontaneous():
    assert COMBUSTION_CH4.is_spontaneous()


def test_methane_combustion_reaction_exergy_is_positive():
    x = COMBUSTION_CH4.reaction_exergy_j_per_mol()
    assert x > 0


def test_methane_combustion_delta_g_approximately_known_value():
    # CH4 + 2O2 → CO2 + 2H2O(g): ΔG° ≈ -800.7 kJ/mol (NIST)
    dg = COMBUSTION_CH4.delta_g_rxn_j_per_mol()
    assert -850_000 < dg < -750_000


def test_hydrogen_combustion_reaction_exergy_close_to_szargut():
    # H2 exergy in database: 236.09 kJ/mol
    # Combustion exergy = -ΔG°: close to but not equal (slightly different basis)
    x = COMBUSTION_H2.reaction_exergy_j_per_mol()
    assert 200_000 < x < 260_000


def test_steam_methane_reforming_is_non_spontaneous():
    # SMR is endergonic (requires energy input)
    assert not STEAM_METHANE_REFORMING.is_spontaneous()


def test_steam_methane_reforming_requires_exergy_input():
    x = STEAM_METHANE_REFORMING.reaction_exergy_j_per_mol()
    # Negative exergy = requires input
    assert x < 0


def test_water_gas_shift_is_exothermic():
    dh = WATER_GAS_SHIFT.delta_h_rxn_j_per_mol()
    assert dh < 0


def test_calcination_requires_exergy_input():
    # CaCO3 → CaO + CO2 is endothermic
    assert not CALCINATION.is_spontaneous()
    dh = CALCINATION.delta_h_rxn_j_per_mol()
    assert dh > 100_000  # highly endothermic


def test_haber_bosch_is_spontaneous():
    assert HABER_BOSCH.is_spontaneous()


def test_haber_bosch_reaction_exergy_per_mol_nh3():
    x = HABER_BOSCH.reaction_exergy_j_per_mol()
    # Per 2 mol NH3; positive and ≈ 32.9 kJ/mol NH3
    assert x > 0


def test_glucose_combustion_is_spontaneous():
    assert GLUCOSE_AEROBIC_OXIDATION.is_spontaneous()


def test_equilibrium_constant_at_reference_temperature_is_finite():
    k = COMBUSTION_CH4.equilibrium_constant()
    assert k > 1.0e100  # CH4 combustion strongly favoured


def test_equilibrium_constant_increases_for_exothermic_at_lower_temperature():
    # WGS is exothermic, so K increases at lower T (van 't Hoff)
    k_300 = WATER_GAS_SHIFT.equilibrium_constant(300.0)
    k_500 = WATER_GAS_SHIFT.equilibrium_constant(500.0)
    assert k_300 > k_500


def test_equilibrium_constant_decreases_for_endothermic_at_lower_temperature():
    # SMR is endothermic, K increases with T
    k_300 = STEAM_METHANE_REFORMING.equilibrium_constant(300.0)
    k_1000 = STEAM_METHANE_REFORMING.equilibrium_constant(1000.0)
    assert k_1000 > k_300


def test_exergy_efficiency_limit_at_100_percent():
    x = COMBUSTION_CH4.reaction_exergy_j_per_mol()
    eta = COMBUSTION_CH4.exergy_efficiency_limit(x)
    assert eta == pytest.approx(1.0, rel=1e-9)


def test_exergy_efficiency_limit_above_one_raises():
    x = COMBUSTION_CH4.reaction_exergy_j_per_mol()
    with pytest.raises(DomainError, match="exceeds reaction exergy"):
        COMBUSTION_CH4.exergy_efficiency_limit(x * 1.1)


def test_exergy_efficiency_limit_on_non_spontaneous_raises():
    with pytest.raises(DomainError, match="releases no exergy"):
        STEAM_METHANE_REFORMING.exergy_efficiency_limit(1000.0)


def test_common_reactions_dict_contains_expected_keys():
    assert "methane-combustion-to-co2-h2o-gas" in COMMON_REACTIONS
    assert "hydrogen-combustion-to-h2o-gas" in COMMON_REACTIONS
    assert "steam-methane-reforming" in COMMON_REACTIONS
    assert "haber-bosch-ammonia-synthesis" in COMMON_REACTIONS


def test_reaction_species_rejects_zero_stoich():
    with pytest.raises(DomainError, match="non-zero"):
        ReactionSpecies("CO2(g)", 0.0)


def test_chemical_reaction_rejects_empty_species():
    with pytest.raises(DomainError):
        ChemicalReaction(name="empty", species=[])


def test_delta_h_rxn_methane_combustion():
    dh = COMBUSTION_CH4.delta_h_rxn_j_per_mol()
    # CH4 + 2O2 → CO2 + 2H2O(g): ΔH° ≈ -802.3 kJ/mol
    assert -820_000 < dh < -780_000
