"""Chemical exergy models: pure substances, combustion, mixing, and reactions.

This package provides the full chemical exergy library for the Exergy
Intelligence Engine.  All models use the Szargut (2005) reference environment.

Public API
──────────
Reference environment:
    ReferenceEnvironment, SZARGUT_RE, R_J_MOL_K, T0_K, P0_PA

Pure substances:
    PureSubstanceRecord, PURE_SUBSTANCE_DB,
    get_by_id, get_by_formula, standard_chemical_exergy_j_per_mol,
    standard_chemical_exergy_j_per_kg

Combustion / beta factors:
    ElementalComposition, GaseousMixtureComponent,
    beta_liquid_fuel, beta_solid_fuel,
    fuel_chemical_exergy_j_per_kg_from_lhv,
    gaseous_mixture_chemical_exergy_j_per_mol,
    gaseous_mixture_lhv_j_per_mol, gaseous_mixture_beta,
    stoichiometric_air_fuel_ratio_kg_kg,
    adiabatic_flame_temperature_estimate_k

Mixing / separation:
    mixing_exergy_j_per_mol, separation_work_j_per_mol_feed,
    separation_work_to_target_j_per_mol_product,
    desalination_min_work_j_per_kg_product,
    co2_capture_min_work_j_per_kg_co2, SeparationTask

Reactions:
    ReactionSpecies, ChemicalReaction,
    COMBUSTION_CH4, COMBUSTION_H2, COMBUSTION_CO,
    STEAM_METHANE_REFORMING, WATER_GAS_SHIFT, CALCINATION,
    HABER_BOSCH, GLUCOSE_AEROBIC_OXIDATION, COMMON_REACTIONS

Models:
    ChemicalExergyModel, PureSubstanceModel,
    BetaFactorLiquidModel, BetaFactorSolidModel,
    GaseousMixtureModel, ReactionExergyModel,
    NATURAL_GAS_UK, BIOGAS_AD, SYNGAS_COAL,
    HYDROGEN_PURE, METHANE_PURE, DIESEL_PROXY, GASOLINE_PROXY,
    ETHANOL_FUEL, BIOMASS_WOOD_PELLET, COMMON_FUEL_MODELS
"""

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
from eie.chemical.mixing import (
    SeparationTask,
    co2_capture_min_work_j_per_kg_co2,
    desalination_min_work_j_per_kg_product,
    mixing_exergy_j_per_mol,
    separation_work_j_per_mol_feed,
    separation_work_to_target_j_per_mol_product,
)
from eie.chemical.models import (
    BIOGAS_AD,
    BIOMASS_WOOD_PELLET,
    COMMON_FUEL_MODELS,
    DIESEL_PROXY,
    ETHANOL_FUEL,
    GASOLINE_PROXY,
    HYDROGEN_PURE,
    METHANE_PURE,
    NATURAL_GAS_UK,
    SYNGAS_COAL,
    BetaFactorLiquidModel,
    BetaFactorSolidModel,
    ChemicalExergyModel,
    GaseousMixtureModel,
    PureSubstanceModel,
    ReactionExergyModel,
)
from eie.chemical.pure_substances import (
    PURE_SUBSTANCE_DB,
    PureSubstanceRecord,
    get_by_formula,
    get_by_id,
    list_substances,
    search_by_formula_prefix,
    standard_chemical_exergy_j_per_kg,
    standard_chemical_exergy_j_per_mol,
)
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
from eie.chemical.reference_environment import (
    P0_PA,
    R_J_MOL_K,
    SZARGUT_RE,
    T0_K,
    ReferenceEnvironment,
    ReferenceSubstance,
)

__all__ = [
    # reference environment
    "P0_PA",
    "R_J_MOL_K",
    "ReferenceEnvironment",
    "ReferenceSubstance",
    "SZARGUT_RE",
    "T0_K",
    # pure substances
    "PURE_SUBSTANCE_DB",
    "PureSubstanceRecord",
    "get_by_formula",
    "get_by_id",
    "list_substances",
    "search_by_formula_prefix",
    "standard_chemical_exergy_j_per_kg",
    "standard_chemical_exergy_j_per_mol",
    # combustion
    "ElementalComposition",
    "GaseousMixtureComponent",
    "adiabatic_flame_temperature_estimate_k",
    "beta_liquid_fuel",
    "beta_solid_fuel",
    "fuel_chemical_exergy_j_per_kg_from_lhv",
    "gaseous_mixture_beta",
    "gaseous_mixture_chemical_exergy_j_per_mol",
    "gaseous_mixture_lhv_j_per_mol",
    "stoichiometric_air_fuel_ratio_kg_kg",
    # mixing
    "SeparationTask",
    "co2_capture_min_work_j_per_kg_co2",
    "desalination_min_work_j_per_kg_product",
    "mixing_exergy_j_per_mol",
    "separation_work_j_per_mol_feed",
    "separation_work_to_target_j_per_mol_product",
    # reactions
    "CALCINATION",
    "COMBUSTION_CH4",
    "COMBUSTION_CO",
    "COMBUSTION_H2",
    "COMMON_REACTIONS",
    "GLUCOSE_AEROBIC_OXIDATION",
    "HABER_BOSCH",
    "STEAM_METHANE_REFORMING",
    "WATER_GAS_SHIFT",
    "ChemicalReaction",
    "ReactionSpecies",
    # models
    "BIOGAS_AD",
    "BIOMASS_WOOD_PELLET",
    "COMMON_FUEL_MODELS",
    "DIESEL_PROXY",
    "ETHANOL_FUEL",
    "GASOLINE_PROXY",
    "HYDROGEN_PURE",
    "METHANE_PURE",
    "NATURAL_GAS_UK",
    "SYNGAS_COAL",
    "BetaFactorLiquidModel",
    "BetaFactorSolidModel",
    "ChemicalExergyModel",
    "GaseousMixtureModel",
    "PureSubstanceModel",
    "ReactionExergyModel",
]
