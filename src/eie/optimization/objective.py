"""Optimization objective definitions and multi-objective aggregation.

Objectives are pure-value descriptors.  Evaluation happens in the evaluator.
All objectives are tagged as "minimize" or "maximize" so that aggregation
and Pareto-dominance comparisons are direction-agnostic.

The module also provides:
• `ObjectiveValue`      — a computed value for one decision
• `MultiObjectiveScore` — collected values for all objectives under one decision
• `pareto_nondominated` — filter to Pareto-nondominated set
• `weighted_sum_score`  — scalar aggregation for ranking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, Sequence

from eie.core.errors import DomainError
from eie.flows.base import require_finite, require_non_negative


Direction = Literal["minimize", "maximize"]


@dataclass(frozen=True)
class OptimizationObjective:
    """Declaration of a single optimization objective.

    Parameters
    ----------
    name : str
        Unique name used as a dictionary key in results.
    direction : "minimize" | "maximize"
        Whether lower or higher values are preferred.
    weight : float
        Relative weight in weighted-sum aggregation (must be >= 0).
        Weights across all active objectives need not sum to 1; normalisation
        is done inside `weighted_sum_score`.
    unit : str
        Physical or currency unit for documentation purposes.
    description : str
        Human-readable description of what this objective measures.
    """

    name: str
    direction: Direction
    weight: float = 1.0
    unit: str = "dimensionless"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise DomainError("OptimizationObjective.name is required")
        if self.direction not in ("minimize", "maximize"):
            raise DomainError(
                f"direction must be 'minimize' or 'maximize', got {self.direction!r}"
            )
        require_non_negative(self.weight, "weight")

    def is_better(self, a: float, b: float) -> bool:
        """Return True if value `a` is better than value `b` for this objective."""
        if self.direction == "minimize":
            return a < b
        return a > b

    def is_at_least_as_good(self, a: float, b: float) -> bool:
        """Return True if value `a` is at least as good as `b`."""
        if self.direction == "minimize":
            return a <= b
        return a >= b

    def normalised_value(self, value: float, worst: float, best: float) -> float:
        """Return value normalised to [0, 1] where 1 = best.

        If best == worst the objective has no variation; returns 0.5.
        """
        require_finite(value, "value")
        if abs(best - worst) < 1.0e-12:
            return 0.5
        if self.direction == "minimize":
            return (worst - value) / (worst - best)
        return (value - worst) / (best - worst)


@dataclass(frozen=True)
class ObjectiveValue:
    """One evaluated objective value for one dispatch candidate."""

    objective_name: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        if not self.objective_name:
            raise DomainError("objective_name is required")
        if not isfinite(self.value):
            raise DomainError("objective value must be finite")


@dataclass(frozen=True)
class MultiObjectiveScore:
    """All objective values for one dispatch candidate."""

    decision_id: str
    values: dict[str, ObjectiveValue] = field(default_factory=dict)

    def get(self, objective_name: str) -> float:
        if objective_name not in self.values:
            raise DomainError(
                f"no objective {objective_name!r} in score for decision {self.decision_id!r}"
            )
        return self.values[objective_name].value

    def objective_names(self) -> list[str]:
        return list(self.values)


def pareto_nondominated(
    scores: Sequence[MultiObjectiveScore],
    objectives: Sequence[OptimizationObjective],
) -> list[MultiObjectiveScore]:
    """Return the Pareto-nondominated subset of candidate scores.

    A candidate A dominates B if:
    - A is at least as good as B on every objective, AND
    - A is strictly better than B on at least one objective.

    The returned list preserves the original order, with dominated solutions
    removed.  O(n²) — suitable for the discrete action spaces in v0.
    """
    if not scores:
        return []
    obj_names = [o.name for o in objectives]
    obj_map = {o.name: o for o in objectives}
    non_dominated: list[MultiObjectiveScore] = []
    for candidate in scores:
        dominated = False
        for other in scores:
            if other.decision_id == candidate.decision_id:
                continue
            other_dominates = True
            other_strictly_better_on_one = False
            for name in obj_names:
                obj = obj_map[name]
                c_val = candidate.get(name)
                o_val = other.get(name)
                if not obj.is_at_least_as_good(o_val, c_val):
                    other_dominates = False
                    break
                if obj.is_better(o_val, c_val):
                    other_strictly_better_on_one = True
            if other_dominates and other_strictly_better_on_one:
                dominated = True
                break
        if not dominated:
            non_dominated.append(candidate)
    return non_dominated


def weighted_sum_score(
    score: MultiObjectiveScore,
    objectives: Sequence[OptimizationObjective],
    worst_values: dict[str, float],
    best_values: dict[str, float],
) -> float:
    """Return a scalar ranking score in [0, 1] (higher = better).

    Each objective is normalised to [0, 1] and then multiplied by its weight.
    The result is the weighted average of normalised scores.

    worst_values and best_values span the full feasible population so that
    normalisation is consistent across all candidates.
    """
    total_weight = sum(o.weight for o in objectives)
    if total_weight <= 0:
        raise DomainError("sum of objective weights must be > 0")
    total = 0.0
    for obj in objectives:
        val = score.get(obj.name)
        worst = worst_values.get(obj.name, val)
        best = best_values.get(obj.name, val)
        normalised = obj.normalised_value(val, worst=worst, best=best)
        total += obj.weight * normalised
    return total / total_weight


# ---------------------------------------------------------------------------
# Standard objective set for the simple-site shadow optimizer
# ---------------------------------------------------------------------------

OBJECTIVE_EXERGY_DESTRUCTION = OptimizationObjective(
    name="exergy_destruction_j",
    direction="minimize",
    weight=2.0,
    unit="J",
    description="Total exergy destroyed in the accounting period",
)

OBJECTIVE_EXERGY_EFFICIENCY = OptimizationObjective(
    name="exergy_efficiency",
    direction="maximize",
    weight=2.0,
    unit="1",
    description="Ratio of useful output exergy to total input exergy",
)

OBJECTIVE_CARBON_KG = OptimizationObjective(
    name="marginal_carbon_kg",
    direction="minimize",
    weight=1.5,
    unit="kg_CO2",
    description="Marginal grid carbon due to any grid import",
)

OBJECTIVE_OPERATING_COST = OptimizationObjective(
    name="operating_cost_currency",
    direction="minimize",
    weight=1.0,
    unit="currency",
    description="Marginal electricity cost for any grid import",
)

OBJECTIVE_BATTERY_SOC = OptimizationObjective(
    name="battery_soc_end",
    direction="maximize",
    weight=0.5,
    unit="1",
    description="Battery state-of-charge at end of period (resilience proxy)",
)

STANDARD_OBJECTIVES: list[OptimizationObjective] = [
    OBJECTIVE_EXERGY_DESTRUCTION,
    OBJECTIVE_EXERGY_EFFICIENCY,
    OBJECTIVE_CARBON_KG,
    OBJECTIVE_OPERATING_COST,
    OBJECTIVE_BATTERY_SOC,
]
