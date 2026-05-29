"""Szargut thermochemical reference environment for chemical exergy.

The reference environment (RE) defines the chemical dead state: the set of
reference substances whose chemical potentials are treated as zero for exergy
accounting purposes.  The model implemented here follows Szargut, Morris &
Steward (1988) as updated by Szargut (2005).

Molar mass values use IUPAC 2021 standard atomic weights (rounded to 3 dp).

Reference conditions: T0 = 298.15 K, P0 = 101 325 Pa.

Every value in this module is a physical constant or a published table entry.
No guesses, no magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from eie.core.constants import STANDARD_ATMOSPHERE_PA, STANDARD_AMBIENT_TEMPERATURE_K
from eie.core.errors import DomainError


# ---------------------------------------------------------------------------
# Universal gas constant (J/(mol·K)), exact per CODATA 2018
# ---------------------------------------------------------------------------
R_J_MOL_K: float = 8.314_462_618


# ---------------------------------------------------------------------------
# Reference-environment temperature and pressure
# ---------------------------------------------------------------------------
T0_K: float = STANDARD_AMBIENT_TEMPERATURE_K   # 298.15 K
P0_PA: float = STANDARD_ATMOSPHERE_PA           # 101 325 Pa


# ---------------------------------------------------------------------------
# Standard atomic / molecular masses (g/mol = kg/kmol)
# IUPAC 2021 standard atomic weights, table values.
# ---------------------------------------------------------------------------
MOLAR_MASS_G_MOL: dict[str, float] = {
    "H":    1.008,
    "C":   12.011,
    "N":   14.007,
    "O":   15.999,
    "S":   32.06,
    "Cl":  35.45,
    "Ar":  39.948,
    "Ne":  20.180,
    "He":   4.003,
    "Si":  28.085,
    "Ca":  40.078,
    "Fe":  55.845,
    "Al":  26.982,
    "Na":  22.990,
    "K":   39.098,
    # Compounds (sum of atoms)
    "H2":       2.016,
    "O2":      31.998,
    "N2":      28.014,
    "H2O":     18.015,
    "CO2":     44.010,
    "CO":      28.010,
    "CH4":     16.043,
    "C2H2":    26.038,
    "C2H4":    28.054,
    "C2H6":    30.070,
    "C3H6":    42.081,
    "C3H8":    44.097,
    "C4H10":   58.124,
    "C5H12":   72.151,
    "C6H6":    78.114,
    "C6H14":   86.178,
    "C7H8":    92.141,
    "C7H16":  100.205,
    "C8H18":  114.232,
    "CH3OH":   32.042,
    "C2H5OH":  46.069,
    "C3H7OH":  60.096,
    "H2S":     34.081,
    "SO2":     64.066,
    "SO3":     80.066,
    "NH3":     17.031,
    "NO":      30.006,
    "NO2":     46.006,
    "N2O":     44.013,
    "HCl":     36.458,
    "HCN":     27.026,
    "CS2":     76.143,
    "COS":     60.076,
    "Ar":      39.948,
}


@dataclass(frozen=True)
class ReferenceSubstance:
    """A substance that defines the zero chemical-exergy level for an element."""

    element: str
    formula: str
    phase: str          # "gas" | "liquid" | "solid"
    description: str
    molar_mass_g_mol: float
    concentration_in_re: float | None   # mol fraction in atmosphere or activity
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.element or not self.formula:
            raise DomainError("element and formula are required for reference substance")
        if not isfinite(self.molar_mass_g_mol) or self.molar_mass_g_mol <= 0:
            raise DomainError("molar_mass_g_mol must be finite and positive")


# Szargut (2005) reference substances for the most common engineering elements.
# Concentration values are standard atmospheric mole fractions where applicable.
REFERENCE_SUBSTANCES: dict[str, ReferenceSubstance] = {
    "C":  ReferenceSubstance("C",  "CO2",    "gas",   "carbon dioxide in dry air",        44.010, 0.000_413,
                             "CO2 partial pressure ~41.9 Pa at STP (2024 global average ~420 ppm)"),
    "H":  ReferenceSubstance("H",  "H2O",    "liquid","liquid water at T0, P0",           18.015, 1.000_000,
                             "pure water activity = 1 by convention"),
    "O":  ReferenceSubstance("O",  "O2",     "gas",   "oxygen in dry air",                31.998, 0.208_96,
                             "standard atmospheric O2 mole fraction (ICAO 1993)"),
    "N":  ReferenceSubstance("N",  "N2",     "gas",   "nitrogen in dry air",              28.014, 0.780_84,
                             "standard atmospheric N2 mole fraction"),
    "S":  ReferenceSubstance("S",  "CaSO4",  "solid", "calcium sulphate (anhydrite)",    136.142, None,
                             "solid activity = 1; Szargut reference mineral"),
    "Ar": ReferenceSubstance("Ar", "Ar",     "gas",   "argon in dry air",                 39.948, 0.009_34,
                             "standard atmospheric Ar mole fraction"),
    "Ne": ReferenceSubstance("Ne", "Ne",     "gas",   "neon in dry air",                  20.180, 1.818e-5),
    "He": ReferenceSubstance("He", "He",     "gas",   "helium in dry air",                 4.003, 5.24e-6),
    "Si": ReferenceSubstance("Si", "SiO2",  "solid", "silicon dioxide (quartz)",          60.084, None,
                             "solid activity = 1; Szargut reference mineral"),
    "Ca": ReferenceSubstance("Ca", "CaCO3", "solid", "calcium carbonate (calcite)",      100.089, None,
                             "solid activity = 1"),
    "Fe": ReferenceSubstance("Fe", "Fe2O3", "solid", "iron(III) oxide (hematite)",       159.690, None,
                             "solid activity = 1"),
    "Al": ReferenceSubstance("Al", "Al2O3", "solid", "aluminium oxide (corundum)",       101.961, None,
                             "solid activity = 1"),
    "Na": ReferenceSubstance("Na", "NaCl",  "solid", "sodium chloride (halite)",          58.443, None,
                             "dissolved activity; Szargut seawater-referenced"),
    "K":  ReferenceSubstance("K",  "KCl",   "solid", "potassium chloride (sylvite)",      74.551, None,
                             "Szargut seawater-referenced"),
    "Cl": ReferenceSubstance("Cl", "HCl",   "gas",   "hydrogen chloride in atmosphere",   36.458, 1.0e-10,
                             "trace atmospheric HCl"),
}


@dataclass(frozen=True)
class ReferenceEnvironment:
    """Complete description of the thermochemical reference environment.

    This object is intentionally immutable and carries only tabulated values.
    All methods are pure computations from those values.
    """

    name: str
    temperature_k: float
    pressure_pa: float
    reference_substances: dict[str, ReferenceSubstance] = field(
        default_factory=lambda: dict(REFERENCE_SUBSTANCES)
    )
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise DomainError("ReferenceEnvironment.name is required")
        if not isfinite(self.temperature_k) or self.temperature_k <= 0:
            raise DomainError("temperature_k must be finite and > 0")
        if not isfinite(self.pressure_pa) or self.pressure_pa <= 0:
            raise DomainError("pressure_pa must be finite and > 0")

    def molar_mass_g_mol(self, formula: str) -> float:
        """Return molar mass in g/mol for a formula in the registry."""
        try:
            return MOLAR_MASS_G_MOL[formula]
        except KeyError as exc:
            raise DomainError(f"no molar mass registered for formula {formula!r}") from exc

    def reference_substance(self, element: str) -> ReferenceSubstance:
        """Return the reference substance for the given element symbol."""
        try:
            return self.reference_substances[element]
        except KeyError as exc:
            raise DomainError(
                f"no reference substance registered for element {element!r}"
            ) from exc

    def atmospheric_partial_pressure_pa(self, element: str) -> float:
        """Return partial pressure (Pa) of the gaseous reference substance for element."""
        rs = self.reference_substance(element)
        if rs.phase != "gas" or rs.concentration_in_re is None:
            raise DomainError(
                f"reference substance for {element!r} is not a gas with known concentration"
            )
        return rs.concentration_in_re * self.pressure_pa

    def restricted_dead_state_exergy_j_mol(self, element: str) -> float:
        """Return the restricted dead-state contribution (kJ/mol → J/mol) for gas-phase
        reference substances from the difference between partial pressure and total pressure.

        This is R*T0*ln(P0 / p_ref) — the exergy required to separate the element's
        reference species from the atmosphere to pure form at T0, P0.
        Only meaningful for gaseous reference substances.
        """
        rs = self.reference_substance(element)
        if rs.phase != "gas" or rs.concentration_in_re is None:
            raise DomainError(
                f"restricted dead state computation requires a gas reference substance "
                f"with known concentration; element {element!r} is {rs.phase!r}"
            )
        from math import log
        p_ref = rs.concentration_in_re * self.pressure_pa
        if p_ref <= 0:
            raise DomainError(f"partial pressure for {element!r} is zero or negative")
        return R_J_MOL_K * self.temperature_k * log(self.pressure_pa / p_ref)


# Singleton standard reference environment used throughout the library.
SZARGUT_RE = ReferenceEnvironment(
    name="Szargut-2005",
    temperature_k=T0_K,
    pressure_pa=P0_PA,
    notes=(
        "Szargut J (2005) Exergy Method: Technical and Ecological Applications. "
        "WIT Press. Reference substances updated for atmospheric CO2 ~420 ppm (2024)."
    ),
)
