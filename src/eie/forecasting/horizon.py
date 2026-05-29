"""Forecast time horizon and step definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import isfinite


@dataclass(frozen=True)
class ForecastStep:
    step_index: int
    start: datetime
    end: datetime
    duration_s: float

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be >= 0")
        if self.end <= self.start:
            raise ValueError("end must be after start")
        if not isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration_s must be finite and > 0")

    @property
    def midpoint(self) -> datetime:
        return self.start + timedelta(seconds=self.duration_s / 2.0)


@dataclass(frozen=True)
class TimeHorizon:
    start: datetime
    step_duration_s: float
    n_steps: int
    steps: tuple[ForecastStep, ...] = field(init=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not isfinite(self.step_duration_s) or self.step_duration_s <= 0:
            raise ValueError("step_duration_s must be finite and > 0")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        delta = timedelta(seconds=self.step_duration_s)
        steps = []
        for i in range(self.n_steps):
            step_start = self.start + i * delta
            steps.append(ForecastStep(step_index=i, start=step_start, end=step_start + delta, duration_s=self.step_duration_s))
        object.__setattr__(self, "steps", tuple(steps))

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.step_duration_s * self.n_steps)

    @property
    def total_duration_s(self) -> float:
        return self.step_duration_s * self.n_steps

    def step_at(self, index: int) -> ForecastStep:
        if index < 0 or index >= self.n_steps:
            raise IndexError(f"step index {index} out of range [0, {self.n_steps - 1}]")
        return self.steps[index]
