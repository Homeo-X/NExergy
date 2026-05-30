"""Strict dimensional unit engine for Exergy Kernel v0.

The kernel keeps the original safety posture: no guessed conversions and no
free-form unit parsing. Only registered units are accepted. Conversions are
scale-only between compatible dimensions, so offset temperature conversions
such as Celsius to Kelvin are intentionally absent.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

from eie.core.errors import OffsetScaleError, UnitError


@dataclass(frozen=True)
class Dimension:
    """Base-dimension exponents for dimensional consistency checks."""

    mass: int = 0
    length: int = 0
    time: int = 0
    temperature: int = 0
    current: int = 0
    currency: int = 0

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(
            self.mass + other.mass,
            self.length + other.length,
            self.time + other.time,
            self.temperature + other.temperature,
            self.current + other.current,
            self.currency + other.currency,
        )

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(
            self.mass - other.mass,
            self.length - other.length,
            self.time - other.time,
            self.temperature - other.temperature,
            self.current - other.current,
            self.currency - other.currency,
        )

    def __pow__(self, exponent: int) -> Dimension:
        return Dimension(
            self.mass * exponent,
            self.length * exponent,
            self.time * exponent,
            self.temperature * exponent,
            self.current * exponent,
            self.currency * exponent,
        )

    @property
    def is_dimensionless(self) -> bool:
        return self == DIMENSIONLESS

    def label(self) -> str:
        parts = []
        for name, exponent in (
            ("M", self.mass),
            ("L", self.length),
            ("T", self.time),
            ("Theta", self.temperature),
            ("I", self.current),
            ("Currency", self.currency),
        ):
            if exponent:
                parts.append(f"{name}^{exponent}")
        return "1" if not parts else " ".join(parts)


DIMENSIONLESS = Dimension()
MASS = Dimension(mass=1)
LENGTH = Dimension(length=1)
TIME = Dimension(time=1)
TEMPERATURE = Dimension(temperature=1)
CURRENT = Dimension(current=1)
CURRENCY = Dimension(currency=1)

ENERGY = MASS * (LENGTH**2) / (TIME**2)
POWER = ENERGY / TIME
PRESSURE = MASS / LENGTH / (TIME**2)
VOLTAGE = POWER / CURRENT
FREQUENCY = DIMENSIONLESS / TIME
MASS_FLOW = MASS / TIME
SPECIFIC_HEAT = ENERGY / MASS / TEMPERATURE
SPECIFIC_ENERGY = ENERGY / MASS
CARBON_INTENSITY = MASS / ENERGY
PRICE_PER_ENERGY = CURRENCY / ENERGY


@dataclass(frozen=True)
class UnitDefinition:
    symbol: str
    dimension: Dimension
    scale_to_si: float
    description: str
    is_offset_scale: bool = False

    def __post_init__(self) -> None:
        if not self.symbol:
            raise UnitError("unit symbol is required")
        if not isfinite(self.scale_to_si) or self.scale_to_si <= 0:
            raise UnitError(f"unit {self.symbol!r} has invalid scale_to_si")


UNIT_REGISTRY: dict[str, UnitDefinition] = {
    "1": UnitDefinition("1", DIMENSIONLESS, 1.0, "dimensionless ratio"),
    "dimensionless": UnitDefinition("dimensionless", DIMENSIONLESS, 1.0, "dimensionless ratio"),
    "kg": UnitDefinition("kg", MASS, 1.0, "kilogram"),
    "g": UnitDefinition("g", MASS, 1.0e-3, "gram"),
    "m": UnitDefinition("m", LENGTH, 1.0, "meter"),
    "s": UnitDefinition("s", TIME, 1.0, "second"),
    "min": UnitDefinition("min", TIME, 60.0, "minute"),
    "h": UnitDefinition("h", TIME, 3_600.0, "hour"),
    "K": UnitDefinition("K", TEMPERATURE, 1.0, "kelvin absolute temperature"),
    # Offset-scale temperatures — interval scales; ratios are physically meaningless.
    # convert_value() refuses these; use AuditedTemperatureConverter instead.
    "°C": UnitDefinition("°C", TEMPERATURE, 1.0, "degree Celsius (offset interval scale)", is_offset_scale=True),
    "degC": UnitDefinition("degC", TEMPERATURE, 1.0, "degree Celsius ASCII alias (offset interval scale)", is_offset_scale=True),
    "°F": UnitDefinition("°F", TEMPERATURE, 5.0 / 9.0, "degree Fahrenheit (offset interval scale)", is_offset_scale=True),
    "degF": UnitDefinition("degF", TEMPERATURE, 5.0 / 9.0, "degree Fahrenheit ASCII alias (offset interval scale)", is_offset_scale=True),
    # Temperature differences — ratio-safe; ΔK = Δ°C, ΔK = Δ°F × 5/9.
    "ΔK": UnitDefinition("ΔK", TEMPERATURE, 1.0, "kelvin temperature difference (ratio-safe)"),
    "Δ°C": UnitDefinition("Δ°C", TEMPERATURE, 1.0, "Celsius temperature difference (ratio-safe)"),
    "Δ°F": UnitDefinition("Δ°F", TEMPERATURE, 5.0 / 9.0, "Fahrenheit temperature difference (ratio-safe)"),
    "A": UnitDefinition("A", CURRENT, 1.0, "ampere"),
    "J": UnitDefinition("J", ENERGY, 1.0, "joule"),
    "kJ": UnitDefinition("kJ", ENERGY, 1.0e3, "kilojoule"),
    "MJ": UnitDefinition("MJ", ENERGY, 1.0e6, "megajoule"),
    "kWh": UnitDefinition("kWh", ENERGY, 3.6e6, "kilowatt-hour"),
    "W": UnitDefinition("W", POWER, 1.0, "watt"),
    "kW": UnitDefinition("kW", POWER, 1.0e3, "kilowatt"),
    "MW": UnitDefinition("MW", POWER, 1.0e6, "megawatt"),
    "Pa": UnitDefinition("Pa", PRESSURE, 1.0, "pascal"),
    "kPa": UnitDefinition("kPa", PRESSURE, 1.0e3, "kilopascal"),
    "bar": UnitDefinition("bar", PRESSURE, 1.0e5, "bar"),
    "V": UnitDefinition("V", VOLTAGE, 1.0, "volt"),
    "Hz": UnitDefinition("Hz", FREQUENCY, 1.0, "hertz"),
    "kg/s": UnitDefinition("kg/s", MASS_FLOW, 1.0, "kilogram per second"),
    "g/s": UnitDefinition("g/s", MASS_FLOW, 1.0e-3, "gram per second"),
    "J/(kg*K)": UnitDefinition("J/(kg*K)", SPECIFIC_HEAT, 1.0, "specific heat capacity"),
    "kJ/(kg*K)": UnitDefinition("kJ/(kg*K)", SPECIFIC_HEAT, 1.0e3, "specific heat capacity"),
    "J/kg": UnitDefinition("J/kg", SPECIFIC_ENERGY, 1.0, "specific energy"),
    "kJ/kg": UnitDefinition("kJ/kg", SPECIFIC_ENERGY, 1.0e3, "specific energy"),
    "kg/kWh": UnitDefinition("kg/kWh", CARBON_INTENSITY, 1.0 / 3.6e6, "mass per energy"),
    "currency/kWh": UnitDefinition("currency/kWh", PRICE_PER_ENERGY, 1.0 / 3.6e6, "currency per energy"),
}

UNIT_DIMENSIONS: dict[str, Dimension] = {
    symbol: definition.dimension for symbol, definition in UNIT_REGISTRY.items()
}


@dataclass(frozen=True)
class Quantity:
    """A numeric value tied to a registered unit."""

    value: float
    unit: str

    def __post_init__(self) -> None:
        if not isfinite(self.value):
            raise UnitError("quantity value must be finite")
        require_known_unit(self.unit)

    @property
    def dimension(self) -> Dimension:
        dimension = dimension_for_unit(self.unit)
        if dimension is None:
            raise UnitError("quantity unit unexpectedly has no dimension")
        return dimension

    def to(self, target_unit: str) -> Quantity:
        return Quantity(convert_value(self.value, self.unit, target_unit), target_unit)

    def require_dimension(self, expected: Dimension, *, field: str = "quantity") -> Quantity:
        require_dimension(self.unit, expected, field=field)
        return self


def unit_is_known(unit: str | None) -> bool:
    if unit is None:
        return True
    return unit in UNIT_REGISTRY


def unit_definition(unit: str | None) -> UnitDefinition | None:
    if unit is None:
        return None
    try:
        return UNIT_REGISTRY[unit]
    except KeyError as exc:
        raise UnitError(f"unknown unit: {unit!r}") from exc


def dimension_for_unit(unit: str | None) -> Dimension | None:
    definition = unit_definition(unit)
    return None if definition is None else definition.dimension


def scale_to_si(unit: str) -> float:
    definition = unit_definition(unit)
    if definition is None:
        raise UnitError("unit is required")
    return definition.scale_to_si


def require_known_unit(unit: str | None) -> None:
    if not unit_is_known(unit):
        raise UnitError(f"unknown unit: {unit!r}")


def require_unit(unit: str | None, allowed_units: Iterable[str], *, field: str = "unit") -> None:
    allowed = set(allowed_units)
    if unit is None:
        raise UnitError(f"{field} is required; expected one of {sorted(allowed)}")
    require_known_unit(unit)
    if unit not in allowed:
        raise UnitError(f"{field} {unit!r} is not compatible; expected one of {sorted(allowed)}")


def require_dimension(unit: str | None, expected: Dimension, *, field: str = "unit") -> None:
    if unit is None:
        raise UnitError(f"{field} is required; expected dimension {expected.label()}")
    actual = dimension_for_unit(unit)
    if actual != expected:
        actual_label = "None" if actual is None else actual.label()
        raise UnitError(
            f"{field} {unit!r} has dimension {actual_label}; expected {expected.label()}"
        )


def require_same_dimension(left_unit: str | None, right_unit: str | None) -> None:
    if left_unit is None or right_unit is None:
        raise UnitError("both units are required for dimensional comparison")
    left_dimension = dimension_for_unit(left_unit)
    right_dimension = dimension_for_unit(right_unit)
    if left_dimension != right_dimension:
        left_label = "None" if left_dimension is None else left_dimension.label()
        right_label = "None" if right_dimension is None else right_dimension.label()
        raise UnitError(
            f"unit dimension mismatch: {left_unit!r} is {left_label}, "
            f"{right_unit!r} is {right_label}"
        )


def compatible_units(left_unit: str, right_unit: str) -> bool:
    return dimension_for_unit(left_unit) == dimension_for_unit(right_unit)


def is_offset_scale_unit(unit: str) -> bool:
    """Return True if unit is an offset-scale temperature (°C, °F)."""
    defn = UNIT_REGISTRY.get(unit)
    return defn is not None and defn.is_offset_scale


def convert_value(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a finite value between registered scale-compatible units.

    Raises OffsetScaleError if either unit is an offset-scale temperature
    (°C, °F).  Use AuditedTemperatureConverter for those conversions.
    """
    if not isfinite(value):
        raise UnitError("value must be finite")
    require_same_dimension(from_unit, to_unit)
    if is_offset_scale_unit(from_unit) or is_offset_scale_unit(to_unit):
        raise OffsetScaleError(
            f"cannot use scale-only convert_value for offset-scale temperature units "
            f"({from_unit!r} → {to_unit!r}); "
            f"use AuditedTemperatureConverter.to_kelvin() instead"
        )
    return value * scale_to_si(from_unit) / scale_to_si(to_unit)


def quantity(value: float, unit: str) -> Quantity:
    return Quantity(value=value, unit=unit)
