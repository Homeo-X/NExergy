"""Standard chemical exergy database for pure substances.

Values are taken from:
  Szargut J, Morris DR, Steward FR (1988) Exergy Analysis of Thermal,
  Chemical, and Metallurgical Processes. Hemisphere, New York.
  Szargut J (2005) Exergy Method: Technical and Ecological Applications.
  WIT Press. Southampton, UK.

All values are given at the standard reference environment:
  T0 = 298.15 K, P0 = 101 325 Pa

Units stored internally: J/mol.
Convenience properties convert to J/kg using registered molar masses.

Phase notation: "g" = ideal gas, "l" = liquid, "s" = solid (crystalline).

The table is intentionally complete for common process-engineering and energy
systems applications. Gaps are documented; no values are guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Final

from eie.chemical.reference_environment import MOLAR_MASS_G_MOL, SZARGUT_RE
from eie.core.errors import DomainError


@dataclass(frozen=True)
class PureSubstanceRecord:
    """Standard chemical exergy data for one pure substance."""

    substance_id: str
    formula: str
    name: str
    phase: str              # "g" | "l" | "s"
    exergy_j_per_mol: float
    molar_mass_g_mol: float
    source_note: str | None = None
    is_reference_substance: bool = False

    def __post_init__(self) -> None:
        if not self.substance_id or not self.formula:
            raise DomainError("substance_id and formula are required")
        if not isfinite(self.exergy_j_per_mol) or self.exergy_j_per_mol < 0:
            raise DomainError(
                f"exergy_j_per_mol must be finite and >= 0 for {self.formula!r}"
            )
        if not isfinite(self.molar_mass_g_mol) or self.molar_mass_g_mol <= 0:
            raise DomainError("molar_mass_g_mol must be finite and > 0")

    @property
    def exergy_j_per_kg(self) -> float:
        """Specific chemical exergy in J/kg."""
        return self.exergy_j_per_mol / (self.molar_mass_g_mol * 1.0e-3)

    @property
    def exergy_kj_per_mol(self) -> float:
        return self.exergy_j_per_mol * 1.0e-3

    @property
    def exergy_kj_per_kg(self) -> float:
        return self.exergy_j_per_kg * 1.0e-3


def _r(
    substance_id: str,
    formula: str,
    name: str,
    phase: str,
    exergy_kj_per_mol: float,
    *,
    molar_mass_override: float | None = None,
    source_note: str | None = None,
    is_reference: bool = False,
) -> PureSubstanceRecord:
    """Convenience constructor — input in kJ/mol, stored as J/mol."""
    mm = molar_mass_override if molar_mass_override is not None else MOLAR_MASS_G_MOL[formula]
    return PureSubstanceRecord(
        substance_id=substance_id,
        formula=formula,
        name=name,
        phase=phase,
        exergy_j_per_mol=exergy_kj_per_mol * 1_000.0,
        molar_mass_g_mol=mm,
        source_note=source_note,
        is_reference_substance=is_reference,
    )


# ---------------------------------------------------------------------------
# The database: all values from Szargut 1988 / 2005 unless noted.
# Sorted by chemical family for readability.
# ---------------------------------------------------------------------------

_DB_RECORDS: list[PureSubstanceRecord] = [
    # ── Simple gases ────────────────────────────────────────────────────────
    _r("O2_g",  "O2",  "oxygen (gas)",       "g",   3.970, source_note="Szargut 2005 p.31"),
    _r("N2_g",  "N2",  "nitrogen (gas)",      "g",   0.720, source_note="Szargut 2005 p.31"),
    _r("Ar_g",  "Ar",  "argon (gas)",         "g",  11.690, source_note="Szargut 2005 p.31"),
    _r("He_g",  "He",  "helium (gas)",        "g",  30.370, source_note="Szargut 2005 p.31"),
    _r("Ne_g",  "Ne",  "neon (gas)",          "g",  27.190, source_note="Szargut 2005 p.31"),

    # ── Hydrogen ────────────────────────────────────────────────────────────
    _r("H2_g",  "H2",  "hydrogen (gas)",      "g", 236.090, source_note="Szargut 2005 p.32"),

    # ── Carbon species ───────────────────────────────────────────────────────
    _r("CO2_g", "CO2", "carbon dioxide (gas)",    "g",  19.480, source_note="Szargut 2005 p.32"),
    _r("CO_g",  "CO",  "carbon monoxide (gas)",   "g", 275.100, source_note="Szargut 2005 p.32"),
    _r("C_s",   "C",   "carbon (graphite/solid)",  "s", 410.260,
       molar_mass_override=12.011, source_note="Szargut 2005 p.32"),

    # ── Water ────────────────────────────────────────────────────────────────
    _r("H2O_l", "H2O", "water (liquid)",      "l",   0.900, source_note="Szargut 2005 p.32",
       is_reference=True),
    _r("H2O_g", "H2O", "water (vapour)",      "g",   9.500, source_note="Szargut 2005 p.32"),

    # ── Light hydrocarbons (gas) ─────────────────────────────────────────────
    _r("CH4_g",   "CH4",   "methane (gas)",         "g",   831.650, source_note="Szargut 2005 p.33"),
    _r("C2H2_g",  "C2H2",  "acetylene (gas)",        "g",  1265.800, source_note="Szargut 2005 p.33"),
    _r("C2H4_g",  "C2H4",  "ethylene (gas)",         "g",  1361.000, source_note="Szargut 2005 p.33"),
    _r("C2H6_g",  "C2H6",  "ethane (gas)",           "g",  1495.800, source_note="Szargut 2005 p.33"),
    _r("C3H6_g",  "C3H6",  "propylene (gas)",        "g",  2003.000, source_note="Szargut 2005 p.33"),
    _r("C3H8_g",  "C3H8",  "propane (gas)",          "g",  2154.000, source_note="Szargut 2005 p.33"),
    _r("C4H10_g", "C4H10", "n-butane (gas)",         "g",  2818.000, source_note="Szargut 2005 p.33"),
    _r("C5H12_g", "C5H12", "n-pentane (gas)",        "g",  3463.000, source_note="Szargut 2005 p.33"),

    # ── Liquid hydrocarbons ──────────────────────────────────────────────────
    _r("C6H6_l",   "C6H6",  "benzene (liquid)",         "l",  3303.000, source_note="Szargut 2005 p.33"),
    _r("C6H14_l",  "C6H14", "n-hexane (liquid)",        "l",  4206.000, source_note="Szargut 2005 p.33"),
    _r("C7H8_l",   "C7H8",  "toluene (liquid)",         "l",  3948.000, source_note="Szargut 2005 p.33"),
    _r("C7H16_l",  "C7H16", "n-heptane (liquid)",       "l",  4862.000, source_note="Szargut 2005 p.33"),
    _r("C8H18_l",  "C8H18", "n-octane (liquid/diesel)", "l",  5413.000, source_note="Szargut 2005 p.33"),

    # ── Alcohols ─────────────────────────────────────────────────────────────
    _r("CH3OH_l",   "CH3OH",  "methanol (liquid)",     "l",   722.300, source_note="Szargut 2005 p.33"),
    _r("C2H5OH_l",  "C2H5OH", "ethanol (liquid)",      "l",  1363.000, source_note="Szargut 2005 p.33"),
    _r("C3H7OH_l",  "C3H7OH", "1-propanol (liquid)",   "l",  2005.000, source_note="Szargut 2005 p.33"),

    # ── Sulphur compounds ────────────────────────────────────────────────────
    _r("H2S_g",  "H2S", "hydrogen sulphide (gas)", "g",  812.000, source_note="Szargut 2005 p.34"),
    _r("SO2_g",  "SO2", "sulphur dioxide (gas)",   "g",  313.400, source_note="Szargut 2005 p.34"),
    _r("SO3_g",  "SO3", "sulphur trioxide (gas)",  "g",  249.100, source_note="Szargut 2005 p.34"),
    _r("CS2_l",  "CS2", "carbon disulphide (liq)", "l", 1065.100, source_note="Szargut 2005 p.34"),
    _r("COS_g",  "COS", "carbonyl sulphide (gas)", "g",  852.400, source_note="Szargut 2005 p.34"),

    # ── Nitrogen compounds ────────────────────────────────────────────────────
    _r("NH3_g",  "NH3", "ammonia (gas)",          "g",  337.900, source_note="Szargut 2005 p.34"),
    _r("NO_g",   "NO",  "nitric oxide (gas)",      "g",   88.900, source_note="Szargut 2005 p.34"),
    _r("NO2_g",  "NO2", "nitrogen dioxide (gas)",  "g",   55.600, source_note="Szargut 2005 p.34"),
    _r("N2O_g",  "N2O", "nitrous oxide (gas)",     "g",  106.890, source_note="Szargut 2005 p.34"),
    _r("HCN_g",  "HCN", "hydrogen cyanide (gas)",  "g",  505.100, source_note="Szargut 2005 p.34"),

    # ── Chlorine compounds ────────────────────────────────────────────────────
    _r("HCl_g",  "HCl", "hydrogen chloride (gas)", "g",   84.500, source_note="Szargut 2005 p.35"),
]

# Build lookup map: substance_id → record
PURE_SUBSTANCE_DB: Final[dict[str, PureSubstanceRecord]] = {r.substance_id: r for r in _DB_RECORDS}

# Secondary lookup: (formula, phase) → first matching record
_FORMULA_PHASE_DB: Final[dict[tuple[str, str], PureSubstanceRecord]] = {
    (r.formula, r.phase): r for r in _DB_RECORDS
}


def get_by_id(substance_id: str) -> PureSubstanceRecord:
    """Return a substance record by its canonical identifier."""
    try:
        return PURE_SUBSTANCE_DB[substance_id]
    except KeyError as exc:
        raise DomainError(
            f"no chemical exergy data for substance_id {substance_id!r}. "
            f"Available: {sorted(PURE_SUBSTANCE_DB)}"
        ) from exc


def get_by_formula(formula: str, phase: str) -> PureSubstanceRecord:
    """Return a substance record by formula and phase ('g', 'l', 's')."""
    key = (formula, phase)
    try:
        return _FORMULA_PHASE_DB[key]
    except KeyError as exc:
        raise DomainError(
            f"no chemical exergy data for formula {formula!r}, phase {phase!r}. "
            f"Available formulas: {sorted({k[0] for k in _FORMULA_PHASE_DB})}"
        ) from exc


def standard_chemical_exergy_j_per_mol(formula: str, phase: str) -> float:
    """Return standard chemical exergy in J/mol."""
    return get_by_formula(formula, phase).exergy_j_per_mol


def standard_chemical_exergy_j_per_kg(formula: str, phase: str) -> float:
    """Return standard chemical exergy in J/kg."""
    return get_by_formula(formula, phase).exergy_j_per_kg


def list_substances() -> list[str]:
    """Return sorted list of registered substance_ids."""
    return sorted(PURE_SUBSTANCE_DB)


def search_by_formula_prefix(prefix: str) -> list[PureSubstanceRecord]:
    """Return all substances whose formula starts with the given prefix."""
    return [r for r in PURE_SUBSTANCE_DB.values() if r.formula.startswith(prefix)]
