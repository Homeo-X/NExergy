"""Tests for optimization objective definitions and multi-objective utilities."""

from __future__ import annotations

import pytest

from eie.core.errors import DomainError
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


def _make_obj(name: str, direction: str, weight: float = 1.0) -> OptimizationObjective:
    return OptimizationObjective(name=name, direction=direction, weight=weight)  # type: ignore[arg-type]


# ── OptimizationObjective validation ────────────────────────────────────────

def test_objective_requires_name():
    with pytest.raises(DomainError):
        OptimizationObjective(name="", direction="minimize")


def test_objective_rejects_invalid_direction():
    with pytest.raises(DomainError):
        OptimizationObjective(name="x", direction="lateral")  # type: ignore[arg-type]


def test_objective_rejects_negative_weight():
    with pytest.raises(DomainError):
        OptimizationObjective(name="x", direction="minimize", weight=-1.0)


def test_objective_is_better_minimize():
    obj = _make_obj("loss", "minimize")
    assert obj.is_better(1.0, 2.0)
    assert not obj.is_better(2.0, 1.0)
    assert not obj.is_better(1.0, 1.0)


def test_objective_is_better_maximize():
    obj = _make_obj("eff", "maximize")
    assert obj.is_better(0.9, 0.8)
    assert not obj.is_better(0.7, 0.8)


def test_objective_is_at_least_as_good():
    obj = _make_obj("loss", "minimize")
    assert obj.is_at_least_as_good(1.0, 2.0)
    assert obj.is_at_least_as_good(1.0, 1.0)
    assert not obj.is_at_least_as_good(2.0, 1.0)


def test_objective_normalised_value_minimize():
    obj = _make_obj("loss", "minimize")
    # worst=10, best=0; value=5 → 0.5
    assert obj.normalised_value(5.0, worst=10.0, best=0.0) == pytest.approx(0.5)
    assert obj.normalised_value(0.0, worst=10.0, best=0.0) == pytest.approx(1.0)
    assert obj.normalised_value(10.0, worst=10.0, best=0.0) == pytest.approx(0.0)


def test_objective_normalised_value_maximize():
    obj = _make_obj("eff", "maximize")
    # worst=0, best=1; value=0.5 → 0.5
    assert obj.normalised_value(0.5, worst=0.0, best=1.0) == pytest.approx(0.5)


def test_objective_normalised_value_no_variation_returns_half():
    obj = _make_obj("loss", "minimize")
    assert obj.normalised_value(5.0, worst=5.0, best=5.0) == pytest.approx(0.5)


# ── ObjectiveValue ──────────────────────────────────────────────────────────

def test_objective_value_requires_name():
    with pytest.raises(DomainError):
        ObjectiveValue(objective_name="", value=1.0, unit="J")


def test_objective_value_rejects_nan():
    with pytest.raises(DomainError):
        ObjectiveValue(objective_name="x", value=float("nan"), unit="J")


# ── MultiObjectiveScore ──────────────────────────────────────────────────────

def _make_score(decision_id: str, **kwargs: float) -> MultiObjectiveScore:
    values = {name: ObjectiveValue(objective_name=name, value=v, unit="1") for name, v in kwargs.items()}
    return MultiObjectiveScore(decision_id=decision_id, values=values)


def test_multi_objective_score_get():
    score = _make_score("d1", loss=10.0, eff=0.8)
    assert score.get("loss") == pytest.approx(10.0)
    assert score.get("eff") == pytest.approx(0.8)


def test_multi_objective_score_missing_objective_raises():
    score = _make_score("d1", loss=10.0)
    with pytest.raises(DomainError):
        score.get("nonexistent")


# ── Pareto-nondominated ──────────────────────────────────────────────────────

def test_pareto_single_candidate_is_non_dominated():
    objectives = [_make_obj("x", "minimize"), _make_obj("y", "maximize")]
    scores = [_make_score("d1", x=1.0, y=0.9)]
    result = pareto_nondominated(scores, objectives)
    assert len(result) == 1


def test_pareto_empty_input_returns_empty():
    result = pareto_nondominated([], [_make_obj("x", "minimize")])
    assert result == []


def test_pareto_dominated_candidate_excluded():
    objectives = [_make_obj("x", "minimize"), _make_obj("y", "maximize")]
    d1 = _make_score("d1", x=1.0, y=0.9)   # best on both: dominates d2
    d2 = _make_score("d2", x=2.0, y=0.7)   # dominated
    result = pareto_nondominated([d1, d2], objectives)
    assert len(result) == 1
    assert result[0].decision_id == "d1"


def test_pareto_trade_off_keeps_both():
    objectives = [_make_obj("x", "minimize"), _make_obj("y", "maximize")]
    d1 = _make_score("d1", x=1.0, y=0.5)   # better x, worse y
    d2 = _make_score("d2", x=2.0, y=0.9)   # worse x, better y
    result = pareto_nondominated([d1, d2], objectives)
    assert len(result) == 2


def test_pareto_three_objectives_finds_frontier():
    objectives = [
        _make_obj("a", "minimize"),
        _make_obj("b", "minimize"),
        _make_obj("c", "maximize"),
    ]
    d1 = _make_score("d1", a=1.0, b=2.0, c=0.9)
    d2 = _make_score("d2", a=2.0, b=1.0, c=0.8)
    d3 = _make_score("d3", a=3.0, b=3.0, c=0.5)  # dominated by d1 and d2
    result = pareto_nondominated([d1, d2, d3], objectives)
    ids = {s.decision_id for s in result}
    assert "d1" in ids
    assert "d2" in ids
    assert "d3" not in ids


# ── weighted_sum_score ───────────────────────────────────────────────────────

def test_weighted_sum_best_candidate_scores_highest():
    objectives = [_make_obj("x", "minimize", weight=1.0), _make_obj("y", "maximize", weight=1.0)]
    best = _make_score("best", x=0.0, y=1.0)
    worst = _make_score("worst", x=10.0, y=0.0)
    worst_vals = {"x": 10.0, "y": 0.0}
    best_vals = {"x": 0.0, "y": 1.0}
    s_best = weighted_sum_score(best, objectives, worst_vals, best_vals)
    s_worst = weighted_sum_score(worst, objectives, worst_vals, best_vals)
    assert s_best > s_worst


def test_weighted_sum_zero_total_weight_raises():
    obj = OptimizationObjective(name="x", direction="minimize", weight=0.0)
    score = _make_score("d1", x=1.0)
    with pytest.raises(DomainError, match="sum of objective weights"):
        weighted_sum_score(score, [obj], {"x": 2.0}, {"x": 0.0})


# ── Standard objectives ──────────────────────────────────────────────────────

def test_standard_objectives_all_valid():
    for obj in STANDARD_OBJECTIVES:
        assert obj.name
        assert obj.direction in ("minimize", "maximize")
        assert obj.weight > 0


def test_standard_objectives_include_exergy_metrics():
    names = {o.name for o in STANDARD_OBJECTIVES}
    assert "exergy_destruction_j" in names
    assert "exergy_efficiency" in names
    assert "marginal_carbon_kg" in names
