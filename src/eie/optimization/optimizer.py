"""Shadow-mode advisory optimizer for the Exergy Intelligence Engine.

The ShadowOptimizer implements a guard-gated discrete-search advisory system.
It operates entirely in shadow mode: it generates advisory dispatch
recommendations only and has no hardware actuation capability.

Algorithm
─────────
1. Grid search: enumerate a discrete grid over the dispatch variable space.
   Default grid: 11 points per variable.
2. Guard evaluation: for each candidate, simulate the exergy outcome and run
   the full guard stack.  Any candidate that fails any guard is discarded —
   there is no "relax constraints" mode.
3. Pareto filtering: compute the Pareto-nondominated frontier from the
   feasible population.
4. Local refinement: for the best weighted-sum candidate on the Pareto front,
   run coordinate descent with 7 bisection steps per variable to refine the
   recommendation.
5. Return an OptimizationResult with the full Pareto front, the refined
   recommendation, per-candidate objective values, and summary statistics.

No candidate that fails any guard is ever included in the output.
The guard_bypass_count is always 0 (enforced by invariant).

Complexity: O(n^k * G) where n = grid points per variable, k = variables,
G = guard evaluation cost.  For k = 3, n = 11: 1331 candidates.
Refinement adds at most 7 * k additional evaluations.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from eie.core.errors import DomainError
from eie.optimization.decision import (
    DispatchDecision,
    DispatchSpace,
    FeasibilityNote,
)
from eie.optimization.evaluator import (
    SimulatedOutcome,
    evaluate_objectives,
    simulate_dispatch,
)
from eie.optimization.objective import (
    MultiObjectiveScore,
    OptimizationObjective,
    STANDARD_OBJECTIVES,
    pareto_nondominated,
    weighted_sum_score,
)
from eie.optimization.snapshot import SiteSnapshot


# ---------------------------------------------------------------------------
# Dispatch space for the simple site (3 variables)
# ---------------------------------------------------------------------------

from eie.optimization.decision import DispatchVariable

SIMPLE_SITE_DISPATCH_SPACE = DispatchSpace(
    variables=[
        DispatchVariable(
            name="heat_pump_fraction",
            lower_bound=0.0,
            upper_bound=1.0,
            unit="1",
            description="Fraction of heat pump rated power to run",
        ),
        DispatchVariable(
            name="battery_charge_fraction",
            lower_bound=0.0,
            upper_bound=1.0,
            unit="1",
            description="Fraction of available PV directed to battery charging",
        ),
        DispatchVariable(
            name="thermal_storage_fraction",
            lower_bound=0.0,
            upper_bound=1.0,
            unit="1",
            description="Fraction of thermal demand served from thermal storage",
        ),
    ],
    coupling_constraints=[
        "heat_pump_fraction * hp_rated_power + battery_charge_fraction * batt_max ≤ pv_available",
        "battery_charge_fraction * batt_max ≤ battery usable charge capacity",
        "thermal_storage_fraction * ts_max_discharge ≤ current thermal demand",
    ],
)


@dataclass(frozen=True)
class OptimizationResult:
    """Complete output of one optimizer run.

    Fields
    ------
    result_id : str
        Unique identifier for this run.
    timestamp : datetime
        When the optimizer was called.
    snapshot_id : str
        The site snapshot this was computed from.
    feasible_decisions : list[DispatchDecision]
        All guard-passing candidates (may be large for dense grids).
    pareto_front : list[DispatchDecision]
        Pareto-nondominated subset of feasible_decisions.
    recommended_decision : DispatchDecision | None
        Best candidate by weighted-sum on the Pareto front (after refinement).
        None if no feasible candidate exists.
    objective_scores : dict[str, MultiObjectiveScore]
        Objective values keyed by decision_id.
    evaluated_count : int
        Total candidates evaluated (including infeasible).
    infeasible_count : int
        Candidates rejected due to guard failures or constraint violations.
    guard_bypass_count : int
        Always 0; invariant enforced by the optimizer.
    objectives : list[OptimizationObjective]
        The objectives used for this run.
    notes : list[str]
        Human-readable diagnostic messages.
    """

    result_id: str
    timestamp: datetime
    snapshot_id: str
    feasible_decisions: list[DispatchDecision]
    pareto_front: list[DispatchDecision]
    recommended_decision: DispatchDecision | None
    objective_scores: dict[str, MultiObjectiveScore]
    evaluated_count: int
    infeasible_count: int
    guard_bypass_count: int
    objectives: list[OptimizationObjective]
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.guard_bypass_count != 0:
            raise DomainError(
                "guard_bypass_count must be 0; the optimizer never bypasses guards"
            )


class ShadowOptimizer:
    """Advisory exergy optimizer — shadow mode, no hardware actuation.

    Parameters
    ----------
    objectives : list[OptimizationObjective]
        Objectives to optimise.  Defaults to STANDARD_OBJECTIVES.
    dispatch_space : DispatchSpace
        Variable bounds and coupling constraints.
        Defaults to SIMPLE_SITE_DISPATCH_SPACE.
    grid_points_per_variable : int
        Grid resolution.  11 gives 1331 candidates for 3 variables.
    refine_steps : int
        Number of bisection steps for local refinement per variable (0 = no refinement).
    """

    def __init__(
        self,
        *,
        objectives: list[OptimizationObjective] | None = None,
        dispatch_space: DispatchSpace | None = None,
        grid_points_per_variable: int = 11,
        refine_steps: int = 7,
    ) -> None:
        self.objectives: list[OptimizationObjective] = (
            objectives if objectives is not None else list(STANDARD_OBJECTIVES)
        )
        self.dispatch_space = (
            dispatch_space if dispatch_space is not None else SIMPLE_SITE_DISPATCH_SPACE
        )
        if grid_points_per_variable < 2:
            raise DomainError("grid_points_per_variable must be >= 2")
        if refine_steps < 0:
            raise DomainError("refine_steps must be >= 0")
        self.grid_points_per_variable = grid_points_per_variable
        self.refine_steps = refine_steps

    def optimise(self, snapshot: SiteSnapshot, *, at: datetime | None = None) -> OptimizationResult:
        """Run the advisory optimisation and return the result.

        Parameters
        ----------
        snapshot : SiteSnapshot
            Complete current state of the site.
        at : datetime | None
            Timestamp for guard freshness checks.  Defaults to snapshot.timestamp.
        """
        timestamp = at if at is not None else snapshot.timestamp
        result_id = str(uuid.uuid4())
        notes: list[str] = []

        # Phase 1: Grid search
        grid = self.dispatch_space.grid_points(self.grid_points_per_variable)
        feasible: list[tuple[DispatchDecision, MultiObjectiveScore]] = []
        infeasible_count = 0
        evaluated_count = 0

        for idx, point in enumerate(grid):
            evaluated_count += 1
            decision_id = f"grid-{idx:05d}"
            outcome = simulate_dispatch(
                snapshot, point, timestamp=timestamp, decision_id=decision_id
            )
            score = evaluate_objectives(outcome, self.objectives)
            fn_notes: list[FeasibilityNote] = []

            if score is None:
                infeasible_count += 1
                failed_guards = [r for r in outcome.guard_results if not r.passed]
                fn_notes.append(
                    FeasibilityNote(
                        code="guard_failure",
                        message="; ".join(r.reason for r in failed_guards),
                        severity="infeasible",
                    )
                )
                decision = DispatchDecision(
                    decision_id=decision_id,
                    timestamp=timestamp,
                    horizon_start=snapshot.horizon_start,
                    horizon_end=snapshot.horizon_end,
                    boundary_id=snapshot.boundary.boundary_id,
                    reference_state_id=snapshot.reference_state.reference_state_id,
                    variable_values=point,
                    is_advisory=True,
                    guard_verified=False,
                    guard_results=outcome.guard_results,
                    feasibility_notes=fn_notes,
                    metadata={},
                )
                continue

            score = MultiObjectiveScore(decision_id=decision_id, values=score.values)
            decision = DispatchDecision(
                decision_id=decision_id,
                timestamp=timestamp,
                horizon_start=snapshot.horizon_start,
                horizon_end=snapshot.horizon_end,
                boundary_id=snapshot.boundary.boundary_id,
                reference_state_id=snapshot.reference_state.reference_state_id,
                variable_values=point,
                is_advisory=True,
                guard_verified=True,
                guard_results=outcome.guard_results,
                feasibility_notes=fn_notes,
                metadata={
                    "pv_exergy_j": outcome.pv_exergy_j,
                    "destroyed_exergy_j": outcome.destroyed_exergy_j,
                    "grid_import_j": outcome.grid_import_j,
                    "battery_soc_end": outcome.battery_soc_end,
                },
            )
            feasible.append((decision, score))

        if not feasible:
            notes.append(
                "No feasible candidates found. All grid points failed guard checks. "
                "Check reference state freshness, boundary binding, and physics parameters."
            )
            return OptimizationResult(
                result_id=result_id,
                timestamp=timestamp,
                snapshot_id=snapshot.snapshot_id,
                feasible_decisions=[],
                pareto_front=[],
                recommended_decision=None,
                objective_scores={},
                evaluated_count=evaluated_count,
                infeasible_count=infeasible_count,
                guard_bypass_count=0,
                objectives=self.objectives,
                notes=notes,
            )

        # Phase 2: Pareto filter
        all_scores = [s for _, s in feasible]
        pareto_scores = pareto_nondominated(all_scores, self.objectives)
        pareto_ids = {s.decision_id for s in pareto_scores}
        feasible_decisions = [d for d, _ in feasible]
        pareto_decisions = [d for d in feasible_decisions if d.decision_id in pareto_ids]

        # Phase 3: Weighted-sum best on Pareto front
        worst_values: dict[str, float] = {}
        best_values: dict[str, float] = {}
        for obj in self.objectives:
            all_vals = [s.get(obj.name) for s in all_scores]
            if obj.direction == "minimize":
                worst_values[obj.name] = max(all_vals)
                best_values[obj.name] = min(all_vals)
            else:
                worst_values[obj.name] = min(all_vals)
                best_values[obj.name] = max(all_vals)

        best_pareto_score = max(
            pareto_scores,
            key=lambda s: weighted_sum_score(s, self.objectives, worst_values, best_values),
        )
        best_pareto_decision = next(
            d for d in pareto_decisions if d.decision_id == best_pareto_score.decision_id
        )

        # Phase 4: Local refinement via coordinate descent + bisection
        if self.refine_steps > 0:
            refined_point, refined_score, refined_outcome = self._refine(
                snapshot,
                best_pareto_decision.variable_values,
                timestamp=timestamp,
                worst_values=worst_values,
                best_values=best_values,
            )
            if refined_score is not None and refined_outcome is not None:
                refined_id = f"refined:{result_id}"
                refined_score = MultiObjectiveScore(
                    decision_id=refined_id, values=refined_score.values
                )
                recommended_decision = DispatchDecision(
                    decision_id=refined_id,
                    timestamp=timestamp,
                    horizon_start=snapshot.horizon_start,
                    horizon_end=snapshot.horizon_end,
                    boundary_id=snapshot.boundary.boundary_id,
                    reference_state_id=snapshot.reference_state.reference_state_id,
                    variable_values=refined_point,
                    is_advisory=True,
                    guard_verified=True,
                    guard_results=refined_outcome.guard_results,
                    feasibility_notes=[],
                    metadata={
                        "refinement": "coordinate_descent_bisection",
                        "refine_steps": self.refine_steps,
                        "pv_exergy_j": refined_outcome.pv_exergy_j,
                        "destroyed_exergy_j": refined_outcome.destroyed_exergy_j,
                    },
                )
                all_scores_with_refined = list(all_scores) + [refined_score]
                all_decisions_with_refined = list(feasible_decisions) + [recommended_decision]
                objective_scores = {
                    s.decision_id: s for s in all_scores_with_refined
                }
            else:
                recommended_decision = best_pareto_decision
                objective_scores = {s.decision_id: s for s in all_scores}
                notes.append("Refinement did not improve the best grid candidate.")
        else:
            recommended_decision = best_pareto_decision
            objective_scores = {s.decision_id: s for s in all_scores}

        notes.append(
            f"Evaluated {evaluated_count} candidates; "
            f"{len(feasible)} feasible; "
            f"{infeasible_count} rejected by guard checks; "
            f"{len(pareto_decisions)} on Pareto front."
        )
        notes.append(
            "All recommendations are advisory only. "
            "Hardware actuation requires an independent safety layer."
        )

        return OptimizationResult(
            result_id=result_id,
            timestamp=timestamp,
            snapshot_id=snapshot.snapshot_id,
            feasible_decisions=feasible_decisions,
            pareto_front=pareto_decisions,
            recommended_decision=recommended_decision,
            objective_scores=objective_scores,
            evaluated_count=evaluated_count,
            infeasible_count=infeasible_count,
            guard_bypass_count=0,
            objectives=self.objectives,
            notes=notes,
        )

    def _refine(
        self,
        snapshot: SiteSnapshot,
        start_point: dict[str, float],
        *,
        timestamp: datetime,
        worst_values: dict[str, float],
        best_values: dict[str, float],
    ) -> tuple[dict[str, float], MultiObjectiveScore | None, SimulatedOutcome | None]:
        """Coordinate-descent refinement with bisection.

        For each variable in turn, perform `refine_steps` bisection steps
        to find the local optimum along that axis.  Returns the refined point,
        its score, and its outcome.
        """
        current_point = dict(start_point)
        current_score: MultiObjectiveScore | None = None
        current_outcome: SimulatedOutcome | None = None

        for _ in range(self.refine_steps):
            improved = False
            for var in self.dispatch_space.variables:
                # Try moving along this axis: split [lower, upper] around current
                lo = var.lower_bound
                hi = var.upper_bound
                for _ in range(3):   # 3 bisection steps per variable per outer iteration
                    mid_lo = (current_point[var.name] + lo) / 2.0
                    mid_hi = (current_point[var.name] + hi) / 2.0
                    best_local = current_point[var.name]
                    best_local_score = current_score
                    best_local_outcome = current_outcome
                    for candidate_val in [mid_lo, mid_hi]:
                        trial = dict(current_point)
                        trial[var.name] = var.clamp(candidate_val)
                        trial_outcome = simulate_dispatch(
                            snapshot, trial, timestamp=timestamp,
                            decision_id=f"refine-{var.name}-{candidate_val:.4f}",
                        )
                        trial_score = evaluate_objectives(trial_outcome, self.objectives)
                        if trial_score is None:
                            continue
                        trial_ws = weighted_sum_score(
                            trial_score, self.objectives, worst_values, best_values
                        )
                        current_ws = (
                            weighted_sum_score(
                                best_local_score, self.objectives, worst_values, best_values
                            )
                            if best_local_score is not None
                            else -1.0
                        )
                        if trial_ws > current_ws:
                            best_local = candidate_val
                            best_local_score = trial_score
                            best_local_outcome = trial_outcome
                    if best_local < current_point[var.name]:
                        hi = current_point[var.name]
                    elif best_local > current_point[var.name]:
                        lo = current_point[var.name]
                    if best_local != current_point[var.name]:
                        current_point[var.name] = var.clamp(best_local)
                        current_score = best_local_score
                        current_outcome = best_local_outcome
                        improved = True
            if not improved:
                break

        # Final evaluation of refined point
        if current_score is None:
            final_outcome = simulate_dispatch(
                snapshot, current_point, timestamp=timestamp, decision_id="refine-final"
            )
            current_score = evaluate_objectives(final_outcome, self.objectives)
            current_outcome = final_outcome

        return current_point, current_score, current_outcome
