"""Shadow-mode advisory optimizer for the Exergy Intelligence Engine.

All optimization output is advisory only.  No hardware actuation.
Guards are always run; any failure makes a candidate infeasible.

Public API
──────────
Objectives:
    OptimizationObjective, ObjectiveValue, MultiObjectiveScore,
    pareto_nondominated, weighted_sum_score, STANDARD_OBJECTIVES,
    OBJECTIVE_EXERGY_DESTRUCTION, OBJECTIVE_EXERGY_EFFICIENCY,
    OBJECTIVE_CARBON_KG, OBJECTIVE_OPERATING_COST, OBJECTIVE_BATTERY_SOC

Decisions:
    DispatchVariable, DispatchSpace, DispatchDecision, FeasibilityNote,
    SIMPLE_SITE_DISPATCH_SPACE

Snapshot:
    SiteSnapshot, BatteryConstraints, ThermalStorageConstraints, GridConstraints

Evaluator:
    SimulatedOutcome, simulate_dispatch, evaluate_objectives

Optimizer:
    ShadowOptimizer, OptimizationResult
"""

from eie.optimization.decision import (
    DispatchDecision,
    DispatchSpace,
    DispatchVariable,
    FeasibilityNote,
)
from eie.optimization.evaluator import (
    SimulatedOutcome,
    evaluate_objectives,
    simulate_dispatch,
)
from eie.optimization.objective import (
    OBJECTIVE_BATTERY_SOC,
    OBJECTIVE_CARBON_KG,
    OBJECTIVE_EXERGY_DESTRUCTION,
    OBJECTIVE_EXERGY_EFFICIENCY,
    OBJECTIVE_OPERATING_COST,
    STANDARD_OBJECTIVES,
    MultiObjectiveScore,
    ObjectiveValue,
    OptimizationObjective,
    pareto_nondominated,
    weighted_sum_score,
)
from eie.optimization.optimizer import (
    SIMPLE_SITE_DISPATCH_SPACE,
    OptimizationResult,
    ShadowOptimizer,
)
from eie.optimization.snapshot import (
    BatteryConstraints,
    GridConstraints,
    SiteSnapshot,
    ThermalStorageConstraints,
)

__all__ = [
    # objectives
    "OBJECTIVE_BATTERY_SOC",
    "OBJECTIVE_CARBON_KG",
    "OBJECTIVE_EXERGY_DESTRUCTION",
    "OBJECTIVE_EXERGY_EFFICIENCY",
    "OBJECTIVE_OPERATING_COST",
    "STANDARD_OBJECTIVES",
    "MultiObjectiveScore",
    "ObjectiveValue",
    "OptimizationObjective",
    "pareto_nondominated",
    "weighted_sum_score",
    # decisions
    "DispatchDecision",
    "DispatchSpace",
    "DispatchVariable",
    "FeasibilityNote",
    "SIMPLE_SITE_DISPATCH_SPACE",
    # snapshot
    "BatteryConstraints",
    "GridConstraints",
    "SiteSnapshot",
    "ThermalStorageConstraints",
    # evaluator
    "SimulatedOutcome",
    "evaluate_objectives",
    "simulate_dispatch",
    # optimizer
    "OptimizationResult",
    "ShadowOptimizer",
]
