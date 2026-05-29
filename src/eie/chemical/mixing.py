"""Mixing and separation exergy for ideal and non-ideal mixtures.

Mixing exergy
─────────────
When streams of different composition are mixed, exergy is destroyed because
the process is irreversible.  Conversely, separating a mixture into pure
streams requires a minimum work equal to the mixing exergy.

For an ideal gas (or ideal solution) mixture:

    Δε_mix = R·T₀·Σᵢ xᵢ·ln(xᵢ)     [J/mol_mixture]

This is always ≤ 0 (negative), meaning mixing releases exergy potential.

Separation work (minimum)
─────────────────────────
To separate a feed mixture (mole fractions xᵢ) into n pure product streams:

    W_sep_min = −R·T₀·Σᵢ xᵢ·ln(xᵢ)  [J/mol_feed]

Separating to enriched (not pure) products with target concentration yᵢ:

    W_sep_min = R·T₀·[Σᵢ yᵢ·ln(yᵢ/xᵢ)]·(n_product / n_feed)

This module also implements:
• Membrane separation minimum work
• Desalination minimum work (seawater model, Van 't Hoff approximation)
• CO₂ capture minimum work (post-combustion)

All equations are isothermal reversible minimum work at T0.
Real systems require more; these are physical lower bounds only.

References
----------
Bejan A, Tsatsaronis G, Moran M (1996) Thermal Design and Optimization.
  Wiley, New York.  Chapter 3.
Moran MJ (1989) Availability Analysis: A Guide to Efficient Energy Use.
  ASME Press.
Szargut J (2005) Exergy Method. WIT Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from typing import Sequence

from eie.chemical.reference_environment import R_J_MOL_K, T0_K
from eie.core.errors import DomainError
from eie.flows.base import require_non_negative, require_positive


def _validate_fractions(fractions: Sequence[float], label: str = "mole_fractions") -> None:
    for i, x in enumerate(fractions):
        if not isfinite(x) or x < 0.0 or x > 1.0 + 1.0e-9:
            raise DomainError(f"{label}[{i}] = {x} is not in [0, 1]")
    total = sum(fractions)
    if abs(total - 1.0) > 1.0e-6:
        raise DomainError(f"{label} sum to {total:.8f}, must equal 1.0 ± 1e-6")


# ---------------------------------------------------------------------------
# Ideal-gas mixing / separation
# ---------------------------------------------------------------------------

def mixing_exergy_j_per_mol(
    mole_fractions: Sequence[float],
    *,
    temperature_k: float = T0_K,
) -> float:
    """Return Gibbs mixing exergy [J/mol] for an ideal mixture.

    The result is always ≤ 0.  Magnitude = minimum separation work.
    Components with xᵢ = 0 contribute nothing (limit: 0·ln(0) = 0).
    """
    require_positive(temperature_k, "temperature_k")
    _validate_fractions(mole_fractions)
    total = sum(
        x * log(x) for x in mole_fractions if x > 0.0
    )
    return R_J_MOL_K * temperature_k * total


def separation_work_j_per_mol_feed(
    feed_fractions: Sequence[float],
    *,
    temperature_k: float = T0_K,
) -> float:
    """Return minimum separation work [J/mol_feed] to separate to pure streams.

    W_sep = −R·T₀·Σᵢ xᵢ·ln(xᵢ) ≥ 0
    """
    return -mixing_exergy_j_per_mol(feed_fractions, temperature_k=temperature_k)


def separation_work_to_target_j_per_mol_product(
    feed_fractions: Sequence[float],
    product_fractions: Sequence[float],
    *,
    temperature_k: float = T0_K,
) -> float:
    """Return minimum work [J/mol_product] to enrich one component from xᵢ to yᵢ.

    Based on the difference in Gibbs free energy between product and feed:
        W_sep = R·T₀·Σᵢ yᵢ·ln(yᵢ/xᵢ)

    This is the Kullback-Leibler divergence (relative entropy) × R·T₀.
    Requires len(feed) == len(product) and both normalised to 1.
    """
    require_positive(temperature_k, "temperature_k")
    _validate_fractions(feed_fractions, "feed_fractions")
    _validate_fractions(product_fractions, "product_fractions")
    if len(feed_fractions) != len(product_fractions):
        raise DomainError("feed_fractions and product_fractions must have equal length")
    total = 0.0
    for x, y in zip(feed_fractions, product_fractions):
        if y <= 0.0:
            continue
        if x <= 0.0:
            raise DomainError(
                "feed fraction is zero for a component that has non-zero product fraction; "
                "separation requires infinite work"
            )
        total += y * log(y / x)
    result = R_J_MOL_K * temperature_k * total
    if result < -1.0e-9:
        raise DomainError(
            f"separation work is negative ({result:.3e}); check feed/product fractions"
        )
    return max(0.0, result)


# ---------------------------------------------------------------------------
# Desalination minimum work (Van 't Hoff / osmotic pressure model)
# ---------------------------------------------------------------------------

#: Approximate osmotic coefficient for seawater at 35 g/kg salinity.
#: Derived from Van 't Hoff: Π = i·C·R·T  ≈ 0.8 MPa at 25°C.
SEAWATER_OSMOTIC_PRESSURE_PA: float = 2.7e6   # 2.7 bar for 35 g/kg NaCl

#: Seawater molar volume [m³/mol] at 298 K (pure water approximation).
WATER_MOLAR_VOLUME_M3_MOL: float = 18.015e-3 / 997.0   # ≈ 1.806e-5 m³/mol

#: Ratio of product water to feed in a once-through RO pass (recovery ratio).
DEFAULT_RO_RECOVERY: float = 0.45


def desalination_min_work_j_per_kg_product(
    *,
    feed_salinity_g_kg: float = 35.0,
    recovery_fraction: float = DEFAULT_RO_RECOVERY,
    temperature_k: float = T0_K,
) -> float:
    """Return minimum specific work [J/kg_product] for desalination.

    Uses the osmotic-pressure reversible-work model (Van 't Hoff / NaCl ideal).
    Actual RO plants require 3–5× this thermodynamic minimum.

    The minimum work integrates the osmotic pressure Π(x) over product volume:

        W_min = (Π₀ / ρ) · (−ln(1 − r)) / r      [J / kg_product]

    where:
    - Π₀ = feed osmotic pressure at 0% recovery (Van 't Hoff: Π = i·m·R·T)
    - ρ  = water density ≈ 997 kg/m³
    - r  = recovery fraction

    At r → 0, W_min → Π₀/ρ (point-osmotic limit, ~2.7 kJ/kg for seawater).
    At r = 0.45, W_min ≈ 3.5–4.5 kJ/kg for 35 g/kg seawater.
    """
    require_positive(feed_salinity_g_kg, "feed_salinity_g_kg")
    require_positive(recovery_fraction, "recovery_fraction")
    if recovery_fraction >= 1.0:
        raise DomainError("recovery_fraction must be < 1")
    require_positive(temperature_k, "temperature_k")

    rho_water = 997.0  # kg/m³ at 25°C
    # NaCl molality [mol/kg_water]: feed_salinity [g/kg_solution] → g/kg_water → mol/kg_water
    #   m = (S_g / M_NaCl) / (1 − S_g/1000)   where S_g is g/kg solution, M_NaCl = 58.44 g/mol
    salt_mol_per_kg_water = feed_salinity_g_kg / 58.44 / (1.0 - feed_salinity_g_kg / 1000.0)

    # Van 't Hoff osmotic pressure [Pa]: Π = i·m·R·T·ρ  (i=2 for NaCl dissociation)
    pi_pa = 2.0 * salt_mol_per_kg_water * R_J_MOL_K * temperature_k * rho_water

    # Minimum specific work integrating over the recovery profile
    # W = (Π₀/ρ) · (−ln(1−r)) / r
    w_min_j_per_kg = (pi_pa / rho_water) * (-log(1.0 - recovery_fraction)) / recovery_fraction
    return w_min_j_per_kg


# ---------------------------------------------------------------------------
# CO₂ capture minimum work
# ---------------------------------------------------------------------------

def co2_capture_min_work_j_per_kg_co2(
    *,
    flue_gas_co2_mole_fraction: float = 0.15,
    capture_fraction: float = 0.90,
    temperature_k: float = T0_K,
) -> float:
    """Return minimum work [J/kg_CO₂] to capture CO₂ from a flue-gas stream.

    Model: isothermal ideal-gas separation from a binary (CO₂ + inerts) stream.

    W_sep = R·T₀·[y·ln(y/x) + (1−y)·ln((1−y)/(1−x))] / (y·M_CO2)

    where x = feed CO₂ mole fraction, y = product purity (≈1 for pure CO₂),
    M_CO₂ = 0.04401 kg/mol.

    This is a strict thermodynamic minimum; real MEA or amine systems
    require 3–6 MJ/kg_CO₂.
    """
    x = flue_gas_co2_mole_fraction
    if not isfinite(x) or not 0.01 <= x <= 0.99:
        raise DomainError("flue_gas_co2_mole_fraction must be in [0.01, 0.99]")
    if not isfinite(capture_fraction) or not 0.01 < capture_fraction < 1.0:
        raise DomainError("capture_fraction must be in (0.01, 1.0)")
    require_positive(temperature_k, "temperature_k")

    # Product stream approximation: essentially pure CO₂ (y → 1)
    # Avoid ln(0) by using y = 0.9999 as upper limit
    y = min(0.9999, capture_fraction)

    # KL divergence per mol feed × R·T₀
    term_co2 = y * log(y / x) if y > 0 and x > 0 else 0.0
    term_inert = (1.0 - y) * log((1.0 - y) / (1.0 - x)) if (1.0 - y) > 0 and (1.0 - x) > 0 else 0.0
    w_j_per_mol_feed = R_J_MOL_K * temperature_k * (term_co2 + term_inert)

    # Convert to J/kg_CO₂: divide by (x * M_CO₂) since x mol CO₂ per mol feed
    M_CO2 = 0.04401  # kg/mol
    w_j_per_kg_co2 = w_j_per_mol_feed / (x * M_CO2)
    return max(0.0, w_j_per_kg_co2)


# ---------------------------------------------------------------------------
# Rich-in-poor-out enrichment model (general membrane / PSA)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeparationTask:
    """Description of a multi-component separation task."""

    name: str
    feed_mole_fractions: list[float]
    product_mole_fractions: list[float]
    component_names: list[str]
    temperature_k: float = T0_K

    def __post_init__(self) -> None:
        if not self.name:
            raise DomainError("SeparationTask.name is required")
        _validate_fractions(self.feed_mole_fractions, "feed_mole_fractions")
        _validate_fractions(self.product_mole_fractions, "product_mole_fractions")
        if len(self.feed_mole_fractions) != len(self.product_mole_fractions):
            raise DomainError("feed and product fraction lists must be equal length")
        if len(self.component_names) != len(self.feed_mole_fractions):
            raise DomainError("component_names must match fraction list length")
        require_positive(self.temperature_k, "temperature_k")

    def minimum_work_j_per_mol_product(self) -> float:
        return separation_work_to_target_j_per_mol_product(
            self.feed_mole_fractions,
            self.product_mole_fractions,
            temperature_k=self.temperature_k,
        )

    def minimum_work_j_per_mol_feed(self) -> float:
        return separation_work_j_per_mol_feed(
            self.feed_mole_fractions,
            temperature_k=self.temperature_k,
        )
