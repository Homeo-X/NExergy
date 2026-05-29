"""Fuel chemical exergy via LHV and Szargut beta factors.

The beta-factor approach (Szargut & Styrylska 1964; Szargut 2005) expresses
chemical exergy of a fuel as:

    ε_ch = β · LHV

where β is a dimensionless ratio that corrects LHV for:
  • the exergy of H₂O produced by combustion (liquid vs. vapour)
  • the exergy contributions of heteroatoms (O, N, S, Cl)
  • molar mass and stoichiometric structure effects

Three distinct β correlations are implemented:
  1. Szargut gaseous fuel β  — compositions of H₂, CH₄, CO, C₂H₆, etc.
  2. Szargut liquid fuel β   — CₙHₘOₖSⱼNᵢ elemental analysis
  3. Szargut solid fuel β    — CₙHₘOₖSⱼNᵢ elemental analysis (Szargut 1988 eq. 4.5)

All correlations are valid for T0 = 298.15 K.

For gaseous mixtures the beta factor is computed rigorously from mole-fraction
weighted standard chemical exergies divided by LHV — no empirical correlation
needed when composition is known.

References
----------
Szargut J, Styrylska T (1964) Approximate evaluation of the exergy of fuels.
  Brennst. Wärme Kraft 16(12):589–596.  [original correlation]
Szargut J, Morris DR, Steward FR (1988) Exergy Analysis of Thermal,
  Chemical, and Metallurgical Processes. Hemisphere, New York.
Szargut J (2005) Exergy Method. WIT Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from eie.chemical.pure_substances import (
    PURE_SUBSTANCE_DB,
    standard_chemical_exergy_j_per_mol,
)
from eie.chemical.reference_environment import MOLAR_MASS_G_MOL, R_J_MOL_K, T0_K
from eie.core.errors import DomainError
from eie.flows.base import require_non_negative, require_positive


# ---------------------------------------------------------------------------
# Standard higher / lower heating values for common pure fuels [J/mol]
# Source: NIST WebBook and Turns "Introduction to Combustion" 3rd ed.
# ---------------------------------------------------------------------------

#: Standard lower heating values (LHV) in J/mol at 298.15 K.
#: LHV = HHV - n_H2O * Δh_vap_H2O where Δh_vap = 44 010 J/mol at 298.15 K
FUEL_LHV_J_PER_MOL: dict[str, float] = {
    "H2":       241_820.0,
    "CH4":      802_300.0,
    "CO":       283_000.0,
    "C2H2":   1_256_000.0,
    "C2H4":   1_323_100.0,
    "C2H6":   1_427_800.0,
    "C3H8":   2_044_000.0,
    "C4H10":  2_658_000.0,
    "C5H12":  3_272_000.0,
    "CH3OH":    638_500.0,
    "C2H5OH": 1_235_400.0,
    "H2S":      518_000.0,
}

#: Standard molar masses for the same fuels [g/mol].
FUEL_MOLAR_MASS: dict[str, float] = {
    k: MOLAR_MASS_G_MOL[k] for k in FUEL_LHV_J_PER_MOL if k in MOLAR_MASS_G_MOL
}


def beta_from_tabulated(formula: str, phase: str = "g") -> float:
    """Return exact β = ε_ch / LHV for a pure fuel with tabulated values.

    Valid for fuels that appear in both the LHV table and the pure-substance DB.
    """
    if formula not in FUEL_LHV_J_PER_MOL:
        raise DomainError(
            f"no tabulated LHV for {formula!r}. "
            f"Available: {sorted(FUEL_LHV_J_PER_MOL)}"
        )
    lhv = FUEL_LHV_J_PER_MOL[formula]
    if lhv <= 0:
        raise DomainError(f"LHV for {formula!r} must be > 0")
    exergy = standard_chemical_exergy_j_per_mol(formula, phase)
    return exergy / lhv


# ---------------------------------------------------------------------------
# Elemental-analysis beta correlations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ElementalComposition:
    """Mass fractions on dry ash-free (daf) basis or molar fractions for gases.

    For liquid/solid fuels: all values are mass fractions [kg/kg], must sum ≤ 1.
    For gaseous mixtures: set is_molar=True and values are mole fractions.
    """

    carbon: float = 0.0        # C
    hydrogen: float = 0.0      # H (not H₂)
    oxygen: float = 0.0        # O (not O₂)
    nitrogen: float = 0.0      # N
    sulphur: float = 0.0       # S
    chlorine: float = 0.0      # Cl
    ash: float = 0.0           # inert mineral mass fraction (solid fuels)
    moisture: float = 0.0      # total moisture mass fraction (solid fuels)
    is_molar: bool = False      # True → values are mole fractions

    def __post_init__(self) -> None:
        for fname in ("carbon", "hydrogen", "oxygen", "nitrogen", "sulphur", "chlorine", "ash", "moisture"):
            v = getattr(self, fname)
            if not isfinite(v) or v < 0.0:
                raise DomainError(f"ElementalComposition.{fname} must be finite and >= 0")
        total = self.carbon + self.hydrogen + self.oxygen + self.nitrogen + self.sulphur + self.chlorine
        if total > 1.0 + 1.0e-6:
            raise DomainError(
                f"elemental mass fractions sum to {total:.6f} > 1; check analysis"
            )

    @property
    def h_c_ratio(self) -> float:
        """Atom ratio H/C (molar) — use only for stoichiometry, NOT for Szargut β."""
        if self.carbon <= 0:
            raise DomainError("H/C ratio undefined when carbon content is zero")
        if not self.is_molar:
            return (self.hydrogen / 1.008) / (self.carbon / 12.011)
        return self.hydrogen / self.carbon

    @property
    def o_c_ratio(self) -> float:
        """Atom ratio O/C (molar)."""
        if self.carbon <= 0:
            return 0.0
        if not self.is_molar:
            return (self.oxygen / 15.999) / (self.carbon / 12.011)
        return self.oxygen / self.carbon

    @property
    def n_c_ratio(self) -> float:
        """Atom ratio N/C (molar)."""
        if self.carbon <= 0:
            return 0.0
        if not self.is_molar:
            return (self.nitrogen / 14.007) / (self.carbon / 12.011)
        return self.nitrogen / self.carbon

    @property
    def s_c_ratio(self) -> float:
        """Atom ratio S/C (molar)."""
        if self.carbon <= 0:
            return 0.0
        if not self.is_molar:
            return (self.sulphur / 32.06) / (self.carbon / 12.011)
        return self.sulphur / self.carbon

    # ── Mass ratios for Szargut β correlations ─────────────────────────────
    # Szargut & Styrylska (1964) and Szargut (2005) use H/C, O/C, S/C, N/C
    # as MASS ratios (kg per kg), not molar atom ratios.

    @property
    def h_c_mass_ratio(self) -> float:
        """Mass ratio H/C [kg_H / kg_C] for Szargut β correlations."""
        if self.carbon <= 0:
            raise DomainError("H/C mass ratio undefined when carbon content is zero")
        if self.is_molar:
            # Convert mole fractions to mass ratio: (H * M_H) / (C * M_C)
            return (self.hydrogen * 1.008) / (self.carbon * 12.011)
        return self.hydrogen / self.carbon

    @property
    def o_c_mass_ratio(self) -> float:
        """Mass ratio O/C [kg_O / kg_C] for Szargut β correlations."""
        if self.carbon <= 0:
            return 0.0
        if self.is_molar:
            return (self.oxygen * 15.999) / (self.carbon * 12.011)
        return self.oxygen / self.carbon

    @property
    def s_c_mass_ratio(self) -> float:
        """Mass ratio S/C [kg_S / kg_C] for Szargut β correlations."""
        if self.carbon <= 0:
            return 0.0
        if self.is_molar:
            return (self.sulphur * 32.06) / (self.carbon * 12.011)
        return self.sulphur / self.carbon

    @property
    def n_c_mass_ratio(self) -> float:
        """Mass ratio N/C [kg_N / kg_C] for Szargut β correlations."""
        if self.carbon <= 0:
            return 0.0
        if self.is_molar:
            return (self.nitrogen * 14.007) / (self.carbon * 12.011)
        return self.nitrogen / self.carbon


def beta_liquid_fuel(composition: ElementalComposition) -> float:
    """Return β for liquid CₙHₘOₖSⱼNᵢ fuels (Szargut 2005, eq. 4.7).

    Valid range (Szargut): 0 ≤ O/C ≤ 0.667, valid for petroleum fractions,
    biodiesel, alcohols.  Correlation uncertainty ±3%.

    The ratios are MASS ratios (kg/kg), per Szargut & Styrylska (1964):
    β = 1.0401 + 0.1728*(H/C) + 0.0432*(O/C)
        + 0.2169*(S/C)*(1 − 2.0628*(H/C))
    """
    hc = composition.h_c_mass_ratio
    oc = composition.o_c_mass_ratio
    sc = composition.s_c_mass_ratio
    beta = (
        1.0401
        + 0.1728 * hc
        + 0.0432 * oc
        + 0.2169 * sc * (1.0 - 2.0628 * hc)
    )
    if not isfinite(beta) or beta <= 0:
        raise DomainError(f"computed liquid beta is non-positive ({beta}); check elemental analysis")
    return beta


def beta_solid_fuel(composition: ElementalComposition) -> float:
    """Return β for solid CₙHₘOₖSⱼNᵢ fuels (Szargut 1988, eq. 4.5).

    Valid for coal, biomass, char, coke, and solid biomass fuels.
    The correlation is on a dry, ash-free (daf) basis.
    Validity: H/C up to ~2.

    The ratios are MASS ratios (kg/kg), per Szargut & Styrylska (1964).
    Corrected form (O/C multiplies the parenthesised bracket):
    β = [1.0438 + 0.1882*(H/C) - 0.2509*(1 + 0.7256*(H/C))*(O/C) + 0.0383*(N/C)]
        / (1 - 0.3035*(O/C))
    """
    hc = composition.h_c_mass_ratio
    oc = composition.o_c_mass_ratio
    nc = composition.n_c_mass_ratio
    denom = 1.0 - 0.3035 * oc
    if denom <= 0:
        raise DomainError(
            "solid fuel beta denominator is zero or negative; O/C ratio too high (>3.30)"
        )
    beta = (1.0438 + 0.1882 * hc - 0.2509 * (1.0 + 0.7256 * hc) * oc + 0.0383 * nc) / denom
    if not isfinite(beta) or beta <= 0:
        raise DomainError(f"computed solid beta is non-positive ({beta}); check elemental analysis")
    return beta


def fuel_chemical_exergy_j_per_kg_from_lhv(
    lhv_j_per_kg: float,
    beta: float,
) -> float:
    """Return specific chemical exergy [J/kg] from LHV and β factor.

    ε_ch = β · LHV
    """
    require_positive(lhv_j_per_kg, "lhv_j_per_kg")
    if not isfinite(beta) or beta <= 0:
        raise DomainError(f"beta must be finite and > 0, got {beta}")
    return beta * lhv_j_per_kg


# ---------------------------------------------------------------------------
# Gaseous mixture chemical exergy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GaseousMixtureComponent:
    """One component of a gaseous fuel mixture."""

    substance_id: str   # key in PURE_SUBSTANCE_DB
    mole_fraction: float

    def __post_init__(self) -> None:
        if not self.substance_id:
            raise DomainError("substance_id is required")
        require_non_negative(self.mole_fraction, "mole_fraction")
        if self.mole_fraction > 1.0 + 1.0e-9:
            raise DomainError("mole_fraction must be <= 1")


def gaseous_mixture_chemical_exergy_j_per_mol(
    components: list[GaseousMixtureComponent],
    *,
    temperature_k: float = T0_K,
    include_mixing_exergy: bool = True,
) -> float:
    """Return chemical exergy of a gaseous fuel mixture per mole of mixture [J/mol].

    This function combines:
    1. Mole-fraction weighted pure-substance exergies.
    2. Ideal-gas mixing exergy: R*T0*Σ(xi*ln(xi)) — always negative (a release),
       so it REDUCES the mixture exergy relative to pure-component sum.
       Include it when the mixture is at its own partial pressures (natural gas
       composition); exclude it when computing the exergy of streams that have
       already been separated.

    The components list must sum to 1 within tolerance.
    """
    require_positive(temperature_k, "temperature_k")
    total_x = sum(c.mole_fraction for c in components)
    if abs(total_x - 1.0) > 1.0e-6:
        raise DomainError(f"mole fractions sum to {total_x:.8f}, must equal 1.0 ± 1e-6")

    from math import log

    weighted_exergy = 0.0
    mixing = 0.0
    for comp in components:
        if comp.mole_fraction <= 0.0:
            continue
        record = PURE_SUBSTANCE_DB[comp.substance_id]
        weighted_exergy += comp.mole_fraction * record.exergy_j_per_mol
        if include_mixing_exergy:
            mixing += comp.mole_fraction * log(comp.mole_fraction)

    return weighted_exergy + (R_J_MOL_K * temperature_k * mixing if include_mixing_exergy else 0.0)


def gaseous_mixture_lhv_j_per_mol(components: list[GaseousMixtureComponent]) -> float:
    """Return LHV of a gaseous mixture [J/mol] by mole-fraction weighting."""
    total_x = sum(c.mole_fraction for c in components)
    if abs(total_x - 1.0) > 1.0e-6:
        raise DomainError(f"mole fractions sum to {total_x:.8f}, must equal 1.0")
    weighted = 0.0
    for comp in components:
        if comp.mole_fraction <= 0.0:
            continue
        record = PURE_SUBSTANCE_DB.get(comp.substance_id)
        if record is None:
            raise DomainError(f"no database record for {comp.substance_id!r}")
        formula = record.formula
        if formula not in FUEL_LHV_J_PER_MOL:
            # Inert (N2, CO2, Ar, O2 in flue gas): LHV contribution = 0
            continue
        weighted += comp.mole_fraction * FUEL_LHV_J_PER_MOL[formula]
    return weighted


def gaseous_mixture_beta(components: list[GaseousMixtureComponent]) -> float:
    """Return β for a gaseous mixture as ε_ch_mix / LHV_mix."""
    lhv = gaseous_mixture_lhv_j_per_mol(components)
    if lhv <= 0:
        raise DomainError("gaseous mixture LHV is zero; cannot compute beta for an inert gas")
    exergy = gaseous_mixture_chemical_exergy_j_per_mol(components, include_mixing_exergy=False)
    return exergy / lhv


# ---------------------------------------------------------------------------
# Combustion stoichiometry helpers
# ---------------------------------------------------------------------------

def stoichiometric_air_fuel_ratio_kg_kg(composition: ElementalComposition) -> float:
    """Return stoichiometric air-fuel ratio [kg_air / kg_fuel] for a solid/liquid fuel.

    Basis: complete combustion to CO₂, H₂O, SO₂, N₂.
    Air is 23.2 mass% O₂.

    O₂ required [kg/kg_fuel] = (32/12)*C + (32/4)*H − O + (32/32)*S
    air required = O₂_req / 0.232
    """
    c = composition.carbon
    h = composition.hydrogen
    o = composition.oxygen
    s = composition.sulphur
    o2_req = (32.0 / 12.011) * c + (32.0 / 4.032) * h - o + (32.0 / 32.06) * s
    if o2_req < 0:
        raise DomainError("stoichiometric O₂ requirement is negative; check oxygen content")
    return o2_req / 0.232


def adiabatic_flame_temperature_estimate_k(
    composition: ElementalComposition,
    lhv_j_per_kg: float,
    *,
    cp_products_j_kg_k: float = 1_150.0,
    t_fuel_k: float = T0_K,
    excess_air_fraction: float = 0.0,
) -> float:
    """Estimate adiabatic flame temperature (K) for a solid/liquid fuel.

    This is an engineering estimate under constant-cp assumption.
    Accurate combustion modelling requires species-resolved equilibrium.

    T_ad = t_fuel + LHV / [(1 + (1+excess)*AFR) * cp_products]
    """
    require_positive(lhv_j_per_kg, "lhv_j_per_kg")
    require_positive(cp_products_j_kg_k, "cp_products_j_kg_k")
    require_positive(t_fuel_k, "t_fuel_k")
    if not isfinite(excess_air_fraction) or excess_air_fraction < 0:
        raise DomainError("excess_air_fraction must be finite and >= 0")
    afr = stoichiometric_air_fuel_ratio_kg_kg(composition)
    mass_ratio = 1.0 + (1.0 + excess_air_fraction) * afr
    delta_t = lhv_j_per_kg / (mass_ratio * cp_products_j_kg_k)
    return t_fuel_k + delta_t
