"""Forecast time horizon and step definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import isfinite


@dataclass(frozen=True)
class ForecastStep:
    """One discrete time step within a forecast horizon.

    Parameters
    ----------
    step_index : int
        Zero-based position within the horizon.
    start : datetime
        Step start time (inclusive).
    end : datetime
        Step end time (exclusive).
    duration_s : float
        Step duration in seconds.
    """

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
        """Midpoint of the step (useful for solar angle calculations)."""
        return self.start + timedelta(seconds=self.duration_s / 2.0)


@dataclass(frozen=True)
class TimeHorizon:
    """A rolling forecast horizon composed of uniform time steps.

    Parameters
    ----------
    start : datetime
        Start of the first step.
    step_duration_s : float
        Duration of each step in seconds.
    n_steps : int
        Number of steps in the horizon.

    Attributes
    ----------
    steps : tuple[ForecastStep, ...]
        All steps in the horizon, computed on construction.
    """

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
            step_end = step_start + delta
            steps.append(
                ForecastStep(
                    step_index=i,
                    start=step_start,
                    end=step_end,
                    duration_s=self.step_duration_s,
                )
            )
        object.__setattr__(self, "steps", tuple(steps))

    @property
    def end(self) -> datetime:
        """End of the last step."""
        return self.start + timedelta(seconds=self.step_duration_s * self.n_steps)

    @property
    def total_duration_s(self) -> float:
        """Total horizon duration in seconds."""
        return self.step_duration_s * self.n_steps

    def step_at(self, index: int) -> ForecastStep:
        """Return the step at *index*.  Raises IndexError for out-of-range."""
        if index < 0 or index >= self.n_steps:
            raise IndexError(
                f"step index {index} out of range [0, {self.n_steps - 1}]"
            )
        return self.steps[index]
