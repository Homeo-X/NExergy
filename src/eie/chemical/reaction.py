"""Chemical reaction exergy.

The exergy transferred in a chemical reaction at constant T₀, P₀ equals the
negative of the standard Gibbs free energy of reaction:

    ε_reaction = −ΔG°_r(T₀, P₀)

For a reaction  Σ νᵢ Aᵢ → Σ νⱼ Bⱼ:
    ΔG°_r = ΔH°_r − T₀·ΔS°_r
          = Σ νⱼ·G°f,Bⱼ − Σ νᵢ·G°f,Aᵢ

Spontaneous reactions (ΔG°_r < 0) release exergy; endergonic reactions
(ΔG°_r > 0) require exergy input.

This module provides:
1. Standard Gibbs free energies of formation for common substances (J/mol).
2. Standard enthalpies of formation (J/mol) for enthalpy-based checks.
3. Standard entropies (J/(mol·K)) for computing ΔS°.
4. A `ChemicalReaction` dataclass for computing reaction exergy.
5. Equilibrium constant K and temperature-dependent corrections via van 't Hoff.

Data source: NIST-JANAF Thermochemical Tables (4th ed., 1998) and NIST WebBook,
all at T = 298.15 K, P = 101 325 Pa.

All values in J/mol.  Gases treated as ideal.

References
----------
Chase MW (1998) NIST-JANAF Thermochemical Tables, 4th ed.  J. Phys. Chem.
  Ref. Data, Monograph 9. National Institute of Standards and Technology.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Final

from eie.chemical.reference_environment import R_J_MOL_K, T0_K
from eie.core.errors import DomainError
from eie.flows.base import require_positive


# ---------------------------------------------------------------------------
# Thermodynamic data at 298.15 K, 101 325 Pa  (all J/mol or J/(mol·K))
# Source: NIST-JANAF 1998.  Zero for elements in their standard state.
# ---------------------------------------------------------------------------

#: Standard enthalpy of formation ΔH°f [J/mol].
DELTA_HF_J_MOL: Final[dict[str, float]] = {
    # Elements (reference, by convention)
    "H2(g)":          0.0,
    "O2(g)":          0.0,
    "N2(g)":          0.0,
    "C(s,graphite)":  0.0,
    "S(s,rhombic)":   0.0,
    "Cl2(g)":         0.0,
    # Inorganic
    "H2O(g)":       -241_826.0,
    "H2O(l)":       -285_830.0,
    "CO(g)":        -110_525.0,
    "CO2(g)":       -393_509.0,
    "NO(g)":          90_291.0,
    "NO2(g)":         33_200.0,
    "N2O(g)":          82_048.0,
    "NH3(g)":         -46_110.0,
    "SO2(g)":       -296_830.0,
    "SO3(g)":       -395_720.0,
    "H2S(g)":        -20_630.0,
    "HCl(g)":        -92_307.0,
    "HNO3(l)":      -174_100.0,
    "H2SO4(l)":     -814_000.0,
    "NaOH(s)":      -425_609.0,
    "CaO(s)":       -635_090.0,
    "CaCO3(s)":    -1_207_600.0,
    # Hydrocarbons
    "CH4(g)":        -74_520.0,
    "C2H2(g)":       226_731.0,
    "C2H4(g)":        52_510.0,
    "C2H6(g)":       -83_820.0,
    "C3H8(g)":      -103_800.0,
    "C4H10(g)":     -126_150.0,
    "C5H12(g)":     -146_440.0,
    "C6H6(l)":        49_000.0,
    "C6H14(l)":     -198_820.0,
    "C7H8(l)":         12_000.0,
    "C8H18(l)":     -250_100.0,
    # Alcohols and oxygenates
    "CH3OH(l)":     -238_700.0,
    "C2H5OH(l)":    -277_690.0,
    "CH3CHO(l)":    -166_190.0,   # acetaldehyde
    "HCOOH(l)":     -424_000.0,   # formic acid
    # Sulphur compounds
    "CS2(l)":         89_700.0,
    "COS(g)":       -142_000.0,
    # Nitrogen compounds
    "HCN(g)":        135_100.0,
    "N2O4(g)":         9_160.0,
    "N2H4(l)":        50_630.0,   # hydrazine
    "CH3NH2(g)":      -22_976.0,  # methylamine
    # Biomass-relevant
    "CH2O(g)":       -108_570.0,  # formaldehyde
    "C12H22O11(s)": -2_222_100.0, # sucrose (representative biomass)
    "C6H12O6(s)":  -1_274_000.0,  # glucose
}

#: Standard Gibbs free energy of formation ΔG°f [J/mol].
DELTA_GF_J_MOL: Final[dict[str, float]] = {
    # Elements (reference)
    "H2(g)":          0.0,
    "O2(g)":          0.0,
    "N2(g)":          0.0,
    "C(s,graphite)":  0.0,
    "S(s,rhombic)":   0.0,
    "Cl2(g)":         0.0,
    # Inorganic
    "H2O(g)":       -228_572.0,
    "H2O(l)":       -237_129.0,
    "CO(g)":        -137_169.0,
    "CO2(g)":       -394_359.0,
    "NO(g)":          86_600.0,
    "NO2(g)":         51_310.0,
    "N2O(g)":         104_200.0,
    "NH3(g)":         -16_450.0,
    "SO2(g)":       -300_194.0,
    "SO3(g)":       -371_060.0,
    "H2S(g)":        -33_450.0,
    "HCl(g)":        -95_299.0,
    "HNO3(l)":      -110_500.0,
    "H2SO4(l)":     -690_003.0,
    "NaOH(s)":      -379_530.0,
    "CaO(s)":       -604_030.0,
    "CaCO3(s)":    -1_128_790.0,
    # Hydrocarbons
    "CH4(g)":        -50_720.0,
    "C2H2(g)":       209_200.0,
    "C2H4(g)":        68_120.0,
    "C2H6(g)":       -31_920.0,
    "C3H8(g)":       -23_470.0,
    "C4H10(g)":       -16_570.0,
    "C5H12(g)":        -8_650.0,
    "C6H6(l)":        124_520.0,
    "C6H14(l)":       -4_000.0,
    "C7H8(l)":         122_000.0,
    "C8H18(l)":         16_530.0,
    # Alcohols
    "CH3OH(l)":     -166_270.0,
    "C2H5OH(l)":    -174_780.0,
    # Sulphur
    "CS2(l)":         65_270.0,
    "COS(g)":       -169_200.0,
    # Nitrogen
    "HCN(g)":        124_700.0,
    "N2H4(l)":        149_340.0,
    # Biomass
    "C6H12O6(s)":   -910_560.0,
    "C12H22O11(s)": -1_544_300.0,
}

#: Standard molar entropy S° [J/(mol·K)].
S0_J_MOL_K: Final[dict[str, float]] = {
    "H2(g)":        130.68,
    "O2(g)":        205.14,
    "N2(g)":        191.61,
    "C(s,graphite)":  5.74,
    "S(s,rhombic)":  31.80,
    "Cl2(g)":       223.08,
    "H2O(g)":       188.83,
    "H2O(l)":        69.91,
    "CO(g)":        197.66,
    "CO2(g)":       213.80,
    "NO(g)":        210.76,
    "NO2(g)":       240.10,
    "N2O(g)":       219.86,
    "NH3(g)":       192.45,
    "SO2(g)":       248.22,
    "SO3(g)":       256.77,
    "H2S(g)":       205.80,
    "HCl(g)":       186.90,
    "CH4(g)":       186.26,
    "C2H2(g)":      200.94,
    "C2H4(g)":      219.56,
    "C2H6(g)":      229.60,
    "C3H8(g)":      270.20,
    "C4H10(g)":     310.23,
    "C6H6(l)":      173.40,
    "CH3OH(l)":     126.80,
    "C2H5OH(l)":    160.70,
    "CaO(s)":        39.75,
    "CaCO3(s)":      91.71,
}


def delta_gf(species: str) -> float:
    """Return ΔG°f [J/mol] for the named species; raises DomainError if unknown."""
    try:
        return DELTA_GF_J_MOL[species]
    except KeyError as exc:
        raise DomainError(
            f"no ΔG°f data for species {species!r}. "
            f"Available: {sorted(DELTA_GF_J_MOL)}"
        ) from exc


def delta_hf(species: str) -> float:
    try:
        return DELTA_HF_J_MOL[species]
    except KeyError as exc:
        raise DomainError(f"no ΔH°f data for species {species!r}") from exc


@dataclass(frozen=True)
class ReactionSpecies:
    """One species in a balanced chemical reaction."""

    name: str       # must exist in DELTA_GF_J_MOL
    stoich: float   # positive for products, negative for reactants

    def __post_init__(self) -> None:
        if not self.name:
            raise DomainError("species name is required")
        if not isfinite(self.stoich) or self.stoich == 0.0:
            raise DomainError("stoichiometric coefficient must be finite and non-zero")


@dataclass(frozen=True)
class ChemicalReaction:
    """A balanced chemical reaction with tabulated thermodynamic data.

    Stoichiometric coefficients: positive for products, negative for reactants.

    Example — methane combustion:
        CH4(g) + 2 O2(g) → CO2(g) + 2 H2O(g)
        species = [
            ReactionSpecies("CH4(g)", -1),
            ReactionSpecies("O2(g)",  -2),
            ReactionSpecies("CO2(g)",  1),
            ReactionSpecies("H2O(g)",  2),
        ]
    """

    name: str
    species: list[ReactionSpecies]
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise DomainError("ChemicalReaction.name is required")
        if not self.species:
            raise DomainError("at least one species is required")

    def delta_g_rxn_j_per_mol(self) -> float:
        """Return ΔG°_r [J/mol] = Σ νᵢ·ΔG°f,i."""
        return sum(s.stoich * delta_gf(s.name) for s in self.species)

    def delta_h_rxn_j_per_mol(self) -> float:
        """Return ΔH°_r [J/mol] = Σ νᵢ·ΔH°f,i."""
        return sum(s.stoich * delta_hf(s.name) for s in self.species)

    def reaction_exergy_j_per_mol(self) -> float:
        """Return chemical exergy released by the reaction [J/mol].

        ε_reaction = −ΔG°_r

        Positive: spontaneous, releases exergy.
        Negative: non-spontaneous, requires exergy input.
        """
        return -self.delta_g_rxn_j_per_mol()

    def equilibrium_constant(self, temperature_k: float = T0_K) -> float:
        """Return equilibrium constant K at the given temperature.

        At T₀:   K₀ = exp(−ΔG°_r / (R·T₀))
        At T:    K(T) ≈ K₀ · exp[−ΔH°_r/R · (1/T − 1/T₀)]  (van 't Hoff)

        This approximation assumes ΔH°_r is temperature-independent (valid for
        small ΔT from T₀; use NASA polynomials for large extrapolations).
        """
        require_positive(temperature_k, "temperature_k")
        delta_g0 = self.delta_g_rxn_j_per_mol()
        k0 = exp(-delta_g0 / (R_J_MOL_K * T0_K))
        if abs(temperature_k - T0_K) < 1.0e-6:
            return k0
        delta_h0 = self.delta_h_rxn_j_per_mol()
        van_t_hoff = exp((-delta_h0 / R_J_MOL_K) * (1.0 / temperature_k - 1.0 / T0_K))
        return k0 * van_t_hoff

    def is_spontaneous(self) -> bool:
        """Return True when ΔG°_r < 0 (reaction releases exergy)."""
        return self.delta_g_rxn_j_per_mol() < 0.0

    def exergy_efficiency_limit(self, useful_exergy_out_j_per_mol: float) -> float:
        """Return exergy efficiency relative to maximum possible output."""
        max_out = self.reaction_exergy_j_per_mol()
        if max_out <= 0:
            raise DomainError(
                "reaction releases no exergy; efficiency relative to reaction exergy is undefined"
            )
        if not isfinite(useful_exergy_out_j_per_mol) or useful_exergy_out_j_per_mol < 0:
            raise DomainError("useful_exergy_out_j_per_mol must be finite and >= 0")
        eta = useful_exergy_out_j_per_mol / max_out
        if eta > 1.0 + 1.0e-9:
            raise DomainError(
                f"useful exergy {useful_exergy_out_j_per_mol:.3e} exceeds reaction exergy "
                f"{max_out:.3e}; physically impossible"
            )
        return eta


# ---------------------------------------------------------------------------
# Pre-defined common reactions
# ---------------------------------------------------------------------------

COMBUSTION_CH4 = ChemicalReaction(
    name="methane-combustion-to-co2-h2o-gas",
    species=[
        ReactionSpecies("CH4(g)", -1),
        ReactionSpecies("O2(g)",  -2),
        ReactionSpecies("CO2(g)",  1),
        ReactionSpecies("H2O(g)",  2),
    ],
    notes="CH4 + 2O2 → CO2 + 2H2O(g); LHV combustion",
)

COMBUSTION_H2 = ChemicalReaction(
    name="hydrogen-combustion-to-h2o-gas",
    species=[
        ReactionSpecies("H2(g)",  -1),
        ReactionSpecies("O2(g)", -0.5),
        ReactionSpecies("H2O(g)", 1),
    ],
    notes="H2 + 0.5 O2 → H2O(g); LHV basis",
)

COMBUSTION_CO = ChemicalReaction(
    name="co-combustion",
    species=[
        ReactionSpecies("CO(g)",  -1),
        ReactionSpecies("O2(g)", -0.5),
        ReactionSpecies("CO2(g)", 1),
    ],
    notes="CO + 0.5 O2 → CO2",
)

STEAM_METHANE_REFORMING = ChemicalReaction(
    name="steam-methane-reforming",
    species=[
        ReactionSpecies("CH4(g)", -1),
        ReactionSpecies("H2O(g)", -1),
        ReactionSpecies("CO(g)",   1),
        ReactionSpecies("H2(g)",   3),
    ],
    notes="SMR: CH4 + H2O → CO + 3H2; endothermic, requires exergy input",
)

WATER_GAS_SHIFT = ChemicalReaction(
    name="water-gas-shift",
    species=[
        ReactionSpecies("CO(g)",   -1),
        ReactionSpecies("H2O(g)",  -1),
        ReactionSpecies("CO2(g)",   1),
        ReactionSpecies("H2(g)",    1),
    ],
    notes="WGS: CO + H2O → CO2 + H2",
)

CALCINATION = ChemicalReaction(
    name="limestone-calcination",
    species=[
        ReactionSpecies("CaCO3(s)", -1),
        ReactionSpecies("CaO(s)",    1),
        ReactionSpecies("CO2(g)",    1),
    ],
    notes="CaCO3 → CaO + CO2; endothermic cement process",
)

HABER_BOSCH = ChemicalReaction(
    name="haber-bosch-ammonia-synthesis",
    species=[
        ReactionSpecies("N2(g)",   -1),
        ReactionSpecies("H2(g)",   -3),
        ReactionSpecies("NH3(g)",   2),
    ],
    notes="N2 + 3H2 → 2NH3; exothermic, high pressure required",
)

GLUCOSE_AEROBIC_OXIDATION = ChemicalReaction(
    name="glucose-aerobic-oxidation",
    species=[
        ReactionSpecies("C6H12O6(s)", -1),
        ReactionSpecies("O2(g)",       -6),
        ReactionSpecies("CO2(g)",       6),
        ReactionSpecies("H2O(l)",       6),
    ],
    notes="C6H12O6 + 6O2 → 6CO2 + 6H2O; cellular respiration / combustion",
)

COMMON_REACTIONS: dict[str, ChemicalReaction] = {
    r.name: r for r in [
        COMBUSTION_CH4,
        COMBUSTION_H2,
        COMBUSTION_CO,
        STEAM_METHANE_REFORMING,
        WATER_GAS_SHIFT,
        CALCINATION,
        HABER_BOSCH,
        GLUCOSE_AEROBIC_OXIDATION,
    ]
}
