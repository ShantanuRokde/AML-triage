"""Leakage-safe per-entity window statistics using sorted composite keys.

Each event is encoded as ``key = entity * SHIFT + t`` (t = integer seconds). After sorting,
all events of one entity are contiguous and time-ordered, so a single ``np.searchsorted``
finds how many events of that entity happened *strictly before* time t. Windows are the
half-open interval ``[t - w, t)``: the current transaction and any other transaction with
the same timestamp are excluded, so a feature at time t only ever sees data from before t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SHIFT = np.int64(2**31)  # > any seconds-since-2000 value until 2068
EPOCH = pd.Timestamp("2000-01-01")


def to_seconds(ts: pd.Series) -> np.ndarray:
    """Convert timestamps to int64 seconds since 2000-01-01 (data-independent epoch)."""
    return ((ts - EPOCH).dt.total_seconds()).to_numpy().astype(np.int64)


@dataclass(frozen=True)
class EventIndex:
    """Sorted composite keys plus prefix sums of an event value (e.g. amount)."""

    keys: np.ndarray
    csum: np.ndarray


def build_index(entity: np.ndarray, t: np.ndarray, values: np.ndarray | None = None) -> EventIndex:
    """Index events by (entity, time). `values` defaults to ones (pure counts)."""
    keys = entity.astype(np.int64) * SHIFT + t.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    vals = np.ones(len(keys)) if values is None else np.asarray(values, dtype=np.float64)
    csum = np.concatenate([[0.0], np.cumsum(vals[order])])
    return EventIndex(keys=keys[order], csum=csum)


def _bounds(index: EventIndex, entity: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (entity base key, first-event position of entity, position of first event >= t)."""
    base = entity.astype(np.int64) * SHIFT
    start = np.searchsorted(index.keys, base, side="left")
    hi = np.searchsorted(index.keys, base + t, side="left")
    return base, start, hi


def count_sum_before(index: EventIndex, entity: np.ndarray, t: np.ndarray,
                     window: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Count and sum of an entity's events in [t - window, t); window=None means all history."""
    base, start, hi = _bounds(index, entity, t)
    if window is None:
        lo = start
    else:
        lo = np.searchsorted(index.keys, base + np.maximum(t - window, 0), side="left")
    return (hi - lo).astype(np.float64), index.csum[hi] - index.csum[lo]


def seconds_since_last(index: EventIndex, entity: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Seconds since the entity's most recent event strictly before t; -1 if none."""
    base, start, hi = _bounds(index, entity, t)
    has = hi > start
    last = index.keys[np.maximum(hi - 1, 0)] - base
    return np.where(has, t - last, -1).astype(np.float64)


def seconds_since_first(index: EventIndex, entity: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Seconds since the entity's first event strictly before t (account age); 0 if none."""
    base, start, hi = _bounds(index, entity, t)
    has = hi > start
    first = index.keys[np.minimum(start, len(index.keys) - 1)] - base
    return np.where(has, t - first, 0).astype(np.float64)
