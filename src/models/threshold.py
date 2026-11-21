"""Decision threshold chosen for an alert budget, not for accuracy."""
from __future__ import annotations

import numpy as np

from src.config import ALERT_BUDGET


def threshold_for_budget(scores: np.ndarray, budget: float = ALERT_BUDGET) -> float:
    """Score cut-off so that `budget` (e.g. 1%) of these transactions are at or above it.

    Choose it on validation scores and apply it unchanged to test, so the test alert rate
    shows how the budget holds up on unseen future data.
    """
    if not 0 < budget < 1:
        raise ValueError("budget must be in (0, 1)")
    return float(np.quantile(np.asarray(scores, dtype=np.float64), 1.0 - budget))
