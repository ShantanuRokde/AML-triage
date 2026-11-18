"""Time-based train/validation/test split. Never random: later data must never train the model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SPLIT_FRACTIONS


def time_split(ts: pd.Series, fractions: tuple[float, float, float] = SPLIT_FRACTIONS) -> tuple[pd.Series, dict[str, str]]:
    """Assign each row to 'train' (earliest 60%), 'val' (next 20%) or 'test' (last 20%).

    Cut points are timestamps, so transactions sharing a timestamp never straddle two splits.
    Requires `ts` to be sorted ascending. Returns (labels, boundary timestamps as strings).
    """
    if not ts.is_monotonic_increasing:
        raise ValueError("time_split requires timestamps sorted ascending")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    n = len(ts)
    t1 = ts.iloc[int(n * fractions[0])]
    t2 = ts.iloc[int(n * (fractions[0] + fractions[1]))]
    labels = np.where(ts < t1, "train", np.where(ts < t2, "val", "test"))
    bounds = {"train_start": str(ts.iloc[0]), "val_start": str(t1), "test_start": str(t2), "test_end": str(ts.iloc[-1])}
    return pd.Series(labels, index=ts.index, name="split"), bounds
