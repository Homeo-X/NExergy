"""Model Predictive Control scheduler for Phase 3.

Provides a rolling-horizon MPC scheduler that optimizes dispatch over a
forecast horizon and returns a complete schedule with gate-verified actions.
Only the first-step actions should be executed; then re-schedule with fresh
site measurements.
"""

from eie.mpc.schedule import MPCSchedule, ScheduledAction
from eie.mpc.scheduler import MPCScheduler

__all__ = [
    "MPCSchedule",
    "MPCScheduler",
    "ScheduledAction",
]
