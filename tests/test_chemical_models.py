"""Tests for chemical exergy model classes and kernel integration."""

from __future__ import annotations

import pytest

from eie.chemical.combustion import ElementalComposition, GaseousMixtureComponent
from eie.chemical.models import (
    BIOGAS_AD,
    BIOMASS_WOOD_PELLET,
    COMMON_FUEL_MODELS,
    DIESEL_PROXY,
    ETHANOL_FUEL,
    HYDROGEN_PURE,
    METHANE_PURE,
    NATURAL_GAS_UK,
    SYNGAS_COAL,
    BetaFactorLiquidModel,
    BetaFactorSolidModel,
    GaseousMixtureModel,
    PureSubstanceModel,
    ReactionExergyModel,
)
from eie.chemical.reaction import COMBUSTION_CH4, COMBUSTION_H2
from eie.core.errors import DomainError
from eie.flows.chemical import ChemicalFlow


# ── PureSubstanceModel ────────────────────────────────────────────────────────

def test_pure_substance_model_hydrogen():
    m = HYDROGEN_PURE
    x = m.specific_chemical_exergy_j_per_kg()
    # ~117 MJ/kg
    assert 110_000_000 < x < 125_000_000


def test_pure_substance_model_methane():
    x = METHANE_PURE.specific_chemical_exergy_j_per_kg()
    # ~51.8 MJ/kg
    assert 48_000_000 < x < 56_000_000


def test_pure_substance_model_diesel_proxy():
    x = DIESEL_PROXY.specific_chemical_exergy_j_per_kg()
    # n-octane ~47.3 MJ/kg
    assert 44_000_000 < x < 52_000_000


def test_pure_substance_model_has_correct_ids():
    assert HYDROGEN_PURE.model_id == "PureSubstanceModel:H2:g"
    assert METHANE_PURE.reference_environment_id == "Szargut-2005"


def test_pure_substance_model_exergy_rate_w():
    x_rate = HYDROGEN_PURE.exergy_rate_w(0.001)  # 1 g/s
    expected = HYDROGEN_PURE.specific_chemical_exergy_j_per_kg() * 0.001
    assert x_rate == pytest.approx(expected, rel=1e-9)


# ── BetaFactorLiquidModel ─────────────────────────────────────────────────────

def test_beta_factor_liquid_model_crude_oil():
    comp = ElementalComposition(carbon=0.853, hydrogen=0.127)
    model = BetaFactorLiquidModel(
        composition=comp, lhv_j_per_kg=42_700_000.0, fuel_name="crude-oil-proxy"
    )
    x = model.specific_chemical_exergy_j_per_kg()
    assert x > 42_700_000.0  # beta > 1
    assert 43_000_000 < x < 48_000_000


def test_beta_factor_liquid_model_requires_positive_lhv():
    comp = ElementalComposition(carbon=0.85, hydrogen=0.12)
    with pytest.raises(DomainError):
        BetaFactorLiquidModel(composition=comp, lhv_j_per_kg=-1.0, fuel_name="bad")


def test_beta_factor_liquid_model_summary():
    comp = ElementalComposition(carbon=0.85, hydrogen=0.12)
    model = BetaFactorLiquidModel(
        composition=comp, lhv_j_per_kg=42_000_000.0, fuel_name="test-fuel"
    )
    summary = model.summary()
    assert "model_id" in summary
    assert "specific_chemical_exergy_j_per_kg" in summary


# ── BetaFactorSolidModel ──────────────────────────────────────────────────────

def test_biomass_wood_pellet_model():
    x = BIOMASS_WOOD_PELLET.specific_chemical_exergy_j_per_kg()
    # ~20 MJ/kg for dry wood with beta ≈ 1.10–1.15
    assert 18_000_000 < x < 25_000_000


def test_beta_factor_solid_model_summary():
    summary = BIOMASS_WOOD_PELLET.summary()
    assert "BetaFactorSolidModel" in summary["model_id"]


# ── GaseousMixtureModel ───────────────────────────────────────────────────────

def test_natural_gas_uk_exergy_per_kg():
    x = NATURAL_GAS_UK.specific_chemical_exergy_j_per_kg()
    # Natural gas: ~47–52 MJ/kg
    assert 44_000_000 < x < 56_000_000


def test_biogas_ad_exergy_per_kg():
    x = BIOGAS_AD.specific_chemical_exergy_j_per_kg()
    # 60% CH4 biogas: ~20–30 MJ/kg
    assert 15_000_000 < x < 35_000_000


def test_syngas_coal_exergy_per_kg():
    x = SYNGAS_COAL.specific_chemical_exergy_j_per_kg()
    assert x > 0


def test_gaseous_mixture_model_beta():
    beta = NATURAL_GAS_UK.beta()
    assert 1.00 < beta < 1.10


def test_gaseous_mixture_model_fractions_must_sum_to_one():
    with pytest.raises(DomainError):
        GaseousMixtureModel(
            components=[
                GaseousMixtureComponent("CH4_g", 0.5),
                GaseousMixtureComponent("H2_g", 0.3),
                # sum = 0.8
            ],
            mixture_name="bad-mixture",
        )


# ── ReactionExergyModel ───────────────────────────────────────────────────────

def test_reaction_exergy_model_methane_combustion():
    model = ReactionExergyModel(
        reaction=COMBUSTION_CH4,
        key_reactant_formula="CH4",
        key_reactant_phase="g",
    )
    x = model.specific_chemical_exergy_j_per_kg()
    # ≈ -ΔG° / M_CH4 ≈ 800kJ/mol / 16g/mol ≈ 50 MJ/kg
    assert 45_000_000 < x < 55_000_000


def test_reaction_exergy_model_hydrogen_combustion():
    model = ReactionExergyModel(
        reaction=COMBUSTION_H2,
        key_reactant_formula="H2",
        key_reactant_phase="g",
    )
    x = model.specific_chemical_exergy_j_per_kg()
    # H2: ~118 MJ/kg
    assert 100_000_000 < x < 130_000_000


# ── COMMON_FUEL_MODELS dict ──────────────────────────────────────────────────

def test_common_fuel_models_dict_contains_expected_keys():
    assert "natural-gas-uk" in COMMON_FUEL_MODELS
    assert "hydrogen-pure" in COMMON_FUEL_MODELS
    assert "biomass-wood-pellet" in COMMON_FUEL_MODELS
    assert "biogas-ad-60-40" in COMMON_FUEL_MODELS


def test_all_common_fuel_models_return_positive_exergy():
    for name, model in COMMON_FUEL_MODELS.items():
        x = model.specific_chemical_exergy_j_per_kg()
        assert x > 0, f"{name} returned non-positive specific exergy"


# ── Kernel integration ────────────────────────────────────────────────────────

def test_kernel_chemical_flow(kernel, boundary, reference, mass_flow_metadata):
    model = METHANE_PURE
    flow = ChemicalFlow(
        flow_id="ng-flow",
        mass_flow_kg_s=0.01,
        specific_chemical_exergy_j_per_kg=model.specific_chemical_exergy_j_per_kg(),
        model_id=model.model_id,
        reference_environment_id=model.reference_environment_id,
        boundary_id=boundary.boundary_id,
        reference_state_id=reference.reference_state_id,
        metadata=mass_flow_metadata,
    )
    exergy_flow = kernel.chemical_flow(flow)
    assert exergy_flow.carrier.value == "chemical"
    assert exergy_flow.exergy_rate_w > 0
    assert exergy_flow.quality_factor == pytest.approx(1.0, rel=1e-9)


def test_kernel_chemical_flow_wrong_boundary_raises(kernel, reference, mass_flow_metadata):
    from eie.core.errors import BoundaryError
    flow = ChemicalFlow(
        flow_id="bad-flow",
        mass_flow_kg_s=0.01,
        specific_chemical_exergy_j_per_kg=50_000_000.0,
        model_id="test-model",
        reference_environment_id="Szargut-2005",
        boundary_id="WRONG-BOUNDARY",
        reference_state_id=reference.reference_state_id,
        metadata=mass_flow_metadata,
    )
    with pytest.raises(BoundaryError):
        kernel.chemical_flow(flow)
