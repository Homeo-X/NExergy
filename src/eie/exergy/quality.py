"""Carrier-specific exergy quality classification."""

from __future__ import annotations

from math import isfinite

from eie.core.enums import Carrier, QualityGrade
from eie.core.errors import DomainError


def validate_quality_factor(quality_factor: float, carrier: Carrier) -> None:
    """Validate qX according to a carrier-specific basis.

    v0 never treats qX = X/E as a universal invariant. For electrical service
    quality the factor is a derating index in [0, 1]. For thermal/cooling and
    chemical carriers the factor is a declared usefulness index and may depend
    on the chosen denominator, so the kernel only requires non-negativity and
    finite value unless a carrier-specific upper bound is physically meaningful.
    """

    if not isfinite(quality_factor):
        raise DomainError("quality_factor must be finite")
    if quality_factor < 0:
        raise DomainError("quality_factor must be >= 0")
    carrier = Carrier(carrier)
    if carrier in {Carrier.ELECTRIC, Carrier.MECHANICAL, Carrier.HYDRAULIC} and quality_factor > 1.0:
        raise DomainError(f"{carrier.value} service quality factor must be <= 1")


def classify_quality(quality_factor: float, carrier: Carrier) -> QualityGrade:
    """Classify usefulness by carrier-specific quality interpretation."""

    carrier = Carrier(carrier)
    validate_quality_factor(quality_factor, carrier)
    if carrier == Carrier.ELECTRIC:
        thresholds = (0.95, 0.80, 0.55, 0.25)
    elif carrier == Carrier.COOLING:
        thresholds = (0.80, 0.55, 0.35, 0.15)
    elif carrier == Carrier.THERMAL:
        thresholds = (0.70, 0.45, 0.20, 0.05)
    elif carrier == Carrier.CHEMICAL:
        thresholds = (0.85, 0.65, 0.40, 0.15)
    else:
        thresholds = (0.90, 0.70, 0.45, 0.20)
    if quality_factor >= thresholds[0]:
        return QualityGrade.A
    if quality_factor >= thresholds[1]:
        return QualityGrade.B
    if quality_factor >= thresholds[2]:
        return QualityGrade.C
    if quality_factor >= thresholds[3]:
        return QualityGrade.D
    return QualityGrade.E
