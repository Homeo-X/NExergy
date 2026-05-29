"""Chemical exergy model classes for use with ChemicalFlow and the kernel.

Every model produces a `specific_chemical_exergy_j_per_kg` value that can
be inserted into a `ChemicalFlow`.  Models document their reference
environment (always SZARGUT_RE) and their assumptions explicitly.

Models available:
──────────────────
1. `PureSubstanceModel`       — direct database lookup (most reliable)
2. `BetaFactorLiquidModel`    — Szargut β for liquid fuels from elemental analysis
3. `BetaFactorSolidModel`     — Szargut β for solid fuels from elemental analysis
4. `GaseousMixtureModel`      — mole-fraction weighted from pure-substance DB
5. `ReactionExergyModel`      — exergy available from a chemical reaction
6. `MultiPhaseModel`          — blends gaseous + liquid phases in a single stream

All models implement the `ChemicalExergyModel` protocol.  This is a plain
abstract-base-class pattern with no runtime overhead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from typing import final

from eie.chemical.combustion import (
    ElementalComposition,
    GaseousMixtureComponent,
    beta_liquid_fuel,
    beta_solid_fuel,
    fuel_chemical_exergy_j_per_kg_from_lhv,
    gaseous_mixture_chemical_exergy_j_per_mol,
    gaseous_mixture_lhv_j_per_mol,
)
from eie.chemical.pure_substances import (
    get_by_formula,
    get_by_id,
    standard_chemical_exergy_j_per_kg,
)
from eie.chemical.reaction import ChemicalReaction
from eie.chemical.reference_environment import MOLAR_MASS_G_MOL, SZARGUT_RE, T0_K, ReferenceEnvironment
from eie.core.errors import DomainError
from eie.flows.base import require_positive


class ChemicalExergyModel(ABC):
    """Abstract base for all chemical exergy models.

    Every concrete model must declare:
    - `model_id`: unique string identifier for use in ChemicalFlow
    - `reference_environment_id`: which RE was used
    - `specific_chemical_exergy_j_per_kg()`: the computed value

    The model does NOT create flows or interact with the kernel; it computes
    the physical number that the user then embeds in a ChemicalFlow.
    """

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def reference_environment_id(self) -> str: ...

    @abstractmethod
    def specific_chemical_exergy_j_per_kg(self) -> float: ...

    def exergy_rate_w(self, mass_flow_kg_s: float) -> float:
        """Return exergy rate [W] for a given mass flow."""
        require_positive(mass_flow_kg_s, "mass_flow_kg_s")
        return mass_flow_kg_s * self.specific_chemical_exergy_j_per_kg()

    def summary(self) -> dict[str, object]:
        """Return a diagnostic dict for logging and ledger annotation."""
        return {
            "model_id": self.model_id,
            "reference_environment_id": self.reference_environment_id,
            "specific_chemical_exergy_j_per_kg": self.specific_chemical_exergy_j_per_kg(),
        }


@final
@dataclass(frozen=True)
class PureSubstanceModel(ChemicalExergyModel):
    """Chemical exergy from the Szargut pure-substance database.

    Parameters
    ----------
    formula : str
        Chemical formula (e.g. "CH4", "H2", "C2H5OH").
    phase : str
        Phase identifier: "g" (gas), "l" (liquid), "s" (solid).
    substance_id : str | None
        Optional override for the database lookup key.
    """

    formula: str
    phase: str
    substance_id_override: str | None = None
    _re: ReferenceEnvironment = SZARGUT_RE

    @property
    def model_id(self) -> str:
        return f"PureSubstanceModel:{self.formula}:{self.phase}"

    @property
    def reference_environment_id(self) -> str:
        return self._re.name

    def specific_chemical_exergy_j_per_kg(self) -> float:
        if self.substance_id_override:
            record = get_by_id(self.substance_id_override)
        else:
            record = get_by_formula(self.formula, self.phase)
        return record.exergy_j_per_kg


@final
@dataclass(frozen=True)
class BetaFactorLiquidModel(ChemicalExergyModel):
    """Chemical exergy for liquid fuels via the Szargut β correlation.

    ε_ch = β_liquid(H/C, O/C, S/C) × LHV

    Accurate for petroleum fractions, fatty acid methyl esters, and alcohols.
    Uncertainty ≈ ±3%.

    Parameters
    ----------
    composition : ElementalComposition
        Elemental mass fractions (dry basis) of the fuel.
    lhv_j_per_kg : float
        Lower heating value of the fuel at 298.15 K [J/kg].
    fuel_name : str
        Human-readable name for documentation.
    """

    composition: ElementalComposition
    lhv_j_per_kg: float
    fuel_name: str

    def __post_init__(self) -> None:
        require_positive(self.lhv_j_per_kg, "lhv_j_per_kg")
        if not self.fuel_name:
            raise DomainError("fuel_name is required")

    @property
    def model_id(self) -> str:
        return f"BetaFactorLiquidModel:{self.fuel_name}"

    @property
    def reference_environment_id(self) -> str:
        return SZARGUT_RE.name

    def beta(self) -> float:
        return beta_liquid_fuel(self.composition)

    def specific_chemical_exergy_j_per_kg(self) -> float:
        return fuel_chemical_exergy_j_per_kg_from_lhv(self.lhv_j_per_kg, self.beta())


@final
@dataclass(frozen=True)
class BetaFactorSolidModel(ChemicalExergyModel):
    """Chemical exergy for solid fuels via the Szargut β correlation.

    ε_ch = β_solid(H/C, O/C, N/C) × LHV

    Accurate for coals, biomass, char, and coke on a dry ash-free basis.
    Uncertainty ≈ ±3–5%.

    Parameters
    ----------
    composition : ElementalComposition
        Elemental mass fractions (daf basis).
    lhv_j_per_kg : float
        Lower heating value on daf basis [J/kg].
    fuel_name : str
        Human-readable name.
    """

    composition: ElementalComposition
    lhv_j_per_kg: float
    fuel_name: str

    def __post_init__(self) -> None:
        require_positive(self.lhv_j_per_kg, "lhv_j_per_kg")
        if not self.fuel_name:
            raise DomainError("fuel_name is required")

    @property
    def model_id(self) -> str:
        return f"BetaFactorSolidModel:{self.fuel_name}"

    @property
    def reference_environment_id(self) -> str:
        return SZARGUT_RE.name

    def beta(self) -> float:
        return beta_solid_fuel(self.composition)

    def specific_chemical_exergy_j_per_kg(self) -> float:
        return fuel_chemical_exergy_j_per_kg_from_lhv(self.lhv_j_per_kg, self.beta())


@final
@dataclass(frozen=True)
class GaseousMixtureModel(ChemicalExergyModel):
    """Chemical exergy of a gaseous fuel mixture from mole-fraction composition.

    For each known fuel component the pure-substance database exergy is used.
    Mixing exergy is optionally included (see include_mixing_exergy).

    The mixture LHV is also computed for diagnostic purposes.
    """

    components: list[GaseousMixtureComponent]
    mixture_name: str
    temperature_k: float = T0_K
    include_mixing_exergy: bool = True

    def __post_init__(self) -> None:
        if not self.mixture_name:
            raise DomainError("mixture_name is required")
        if not self.components:
            raise DomainError("at least one component is required")
        require_positive(self.temperature_k, "temperature_k")
        total_x = sum(c.mole_fraction for c in self.components)
        if abs(total_x - 1.0) > 1.0e-6:
            raise DomainError(
                f"GaseousMixtureModel '{self.mixture_name}': component mole fractions sum to "
                f"{total_x:.8f}, must equal 1.0 ± 1e-6"
            )

    @property
    def model_id(self) -> str:
        return f"GaseousMixtureModel:{self.mixture_name}"

    @property
    def reference_environment_id(self) -> str:
        return SZARGUT_RE.name

    def mixture_exergy_j_per_mol(self) -> float:
        return gaseous_mixture_chemical_exergy_j_per_mol(
            self.components,
            temperature_k=self.temperature_k,
            include_mixing_exergy=self.include_mixing_exergy,
        )

    def mixture_lhv_j_per_mol(self) -> float:
        return gaseous_mixture_lhv_j_per_mol(self.components)

    def mixture_molar_mass_g_mol(self) -> float:
        from eie.chemical.pure_substances import PURE_SUBSTANCE_DB
        total = 0.0
        for comp in self.components:
            rec = PURE_SUBSTANCE_DB[comp.substance_id]
            total += comp.mole_fraction * rec.molar_mass_g_mol
        return total

    def specific_chemical_exergy_j_per_kg(self) -> float:
        mm = self.mixture_molar_mass_g_mol()
        if mm <= 0:
            raise DomainError("computed mixture molar mass is zero or negative")
        return self.mixture_exergy_j_per_mol() / (mm * 1.0e-3)

    def beta(self) -> float:
        """Return β = ε_ch_mix / LHV_mix for this gaseous mixture."""
        lhv = self.mixture_lhv_j_per_mol()
        if lhv <= 0:
            raise DomainError("mixture LHV is zero; beta undefined for inert mixture")
        return self.mixture_exergy_j_per_mol() / lhv


@final
@dataclass(frozen=True)
class ReactionExergyModel(ChemicalExergyModel):
    """Chemical exergy available from a balanced chemical reaction.

    This model is appropriate when the stream drives a chemical conversion
    (e.g. fuel cell, reactor) and the exergy of interest is the maximum
    reversible work obtainable from the reaction, not the specific chemical
    exergy of a single fuel component.

    specific_chemical_exergy_j_per_kg is computed as:
        ε_reaction [J/mol] / M_reactant_key [kg/mol]

    where M_reactant_key is the molar mass of the nominated key reactant species.
    """

    reaction: ChemicalReaction
    key_reactant_formula: str       # e.g. "CH4" for methane combustion
    key_reactant_phase: str = "g"

    def __post_init__(self) -> None:
        if not self.key_reactant_formula:
            raise DomainError("key_reactant_formula is required")

    @property
    def model_id(self) -> str:
        return f"ReactionExergyModel:{self.reaction.name}:{self.key_reactant_formula}"

    @property
    def reference_environment_id(self) -> str:
        return SZARGUT_RE.name

    def reaction_exergy_j_per_mol(self) -> float:
        return self.reaction.reaction_exergy_j_per_mol()

    def specific_chemical_exergy_j_per_kg(self) -> float:
        record = get_by_formula(self.key_reactant_formula, self.key_reactant_phase)
        return self.reaction_exergy_j_per_mol() / (record.molar_mass_g_mol * 1.0e-3)


# ---------------------------------------------------------------------------
# Pre-built model instances for the most common engineering fuels
# ---------------------------------------------------------------------------

#: Natural gas composition (UK North Sea approximate, 2023 average)
NATURAL_GAS_UK = GaseousMixtureModel(
    components=[
        GaseousMixtureComponent("CH4_g",   0.890),
        GaseousMixtureComponent("C2H6_g",  0.060),
        GaseousMixtureComponent("C3H8_g",  0.020),
        GaseousMixtureComponent("C4H10_g", 0.005),
        GaseousMixtureComponent("N2_g",    0.015),
        GaseousMixtureComponent("CO2_g",   0.010),
    ],
    mixture_name="natural-gas-uk",
    include_mixing_exergy=True,
)

#: Biogas from anaerobic digestion (~60% CH4, 40% CO2)
BIOGAS_AD = GaseousMixtureModel(
    components=[
        GaseousMixtureComponent("CH4_g",  0.60),
        GaseousMixtureComponent("CO2_g",  0.40),
    ],
    mixture_name="biogas-ad-60-40",
    include_mixing_exergy=True,
)

#: Syngas from coal gasification (generic, molar composition)
SYNGAS_COAL = GaseousMixtureModel(
    components=[
        GaseousMixtureComponent("H2_g",   0.30),
        GaseousMixtureComponent("CO_g",   0.45),
        GaseousMixtureComponent("CO2_g",  0.10),
        GaseousMixtureComponent("CH4_g",  0.05),
        GaseousMixtureComponent("N2_g",   0.10),
    ],
    mixture_name="syngas-coal-generic",
    include_mixing_exergy=True,
)

#: Hydrogen (pure, gaseous)
HYDROGEN_PURE = PureSubstanceModel(formula="H2", phase="g")

#: Methane (pure, gaseous) — natural-gas surrogate
METHANE_PURE = PureSubstanceModel(formula="CH4", phase="g")

#: Diesel fuel proxy — n-octane liquid
DIESEL_PROXY = PureSubstanceModel(formula="C8H18", phase="l")

#: Gasoline fuel proxy — n-heptane liquid
GASOLINE_PROXY = PureSubstanceModel(formula="C7H16", phase="l")

#: Ethanol fuel (E100) — liquid
ETHANOL_FUEL = PureSubstanceModel(formula="C2H5OH", phase="l")

#: Biomass (wood pellet representative — dry pine, daf basis)
BIOMASS_WOOD_PELLET = BetaFactorSolidModel(
    composition=ElementalComposition(
        carbon=0.511,
        hydrogen=0.060,
        oxygen=0.421,
        nitrogen=0.003,
        sulphur=0.001,
        ash=0.004,
    ),
    lhv_j_per_kg=18_500_000.0,   # 18.5 MJ/kg dry
    fuel_name="wood-pellet-pine-daf",
)

COMMON_FUEL_MODELS: dict[str, ChemicalExergyModel] = {
    "natural-gas-uk":        NATURAL_GAS_UK,
    "biogas-ad-60-40":       BIOGAS_AD,
    "syngas-coal-generic":   SYNGAS_COAL,
    "hydrogen-pure":         HYDROGEN_PURE,
    "methane-pure":          METHANE_PURE,
    "diesel-proxy":          DIESEL_PROXY,
    "gasoline-proxy":        GASOLINE_PROXY,
    "ethanol-fuel":          ETHANOL_FUEL,
    "biomass-wood-pellet":   BIOMASS_WOOD_PELLET,
}
