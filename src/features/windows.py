"""Leakage-safe per-entity window statistics using sorted composite keys.

Each event is encoded as ``key = entity * SHIFT + t`` (t = integer seconds). After sorting,
all events of one entity are contiguous and time-ordered, so a single ``np.searchsorted``
finds how many events of that entity happened *strictly before* time t. Windows are the
half-open interval ``[t - w, t)``: the current transaction and any other transaction with
the same timestamp are excluded, so a feature at time t only ever sees data from before t.

Performance: queries are sorted once per (entity, t) array (`Queries`) and that order is
reused for every window, because ``entity*SHIFT + max(t - w, 0)`` preserves the (entity, t)
order for any w. Binary searches over sorted needles walk memory in order (cache friendly).
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


@dataclass(frozen=True)
class Queries:
    """Query points (entity, t) with their sort order precomputed once."""

    t: np.ndarray
    base: np.ndarray
    order: np.ndarray

    @classmethod
    def make(cls, entity: np.ndarray, t: np.ndarray) -> "Queries":
        """Build queries for the given entity codes and int64 second timestamps."""
        base = entity.astype(np.int64) * SHIFT
        return cls(t=t.astype(np.int64), base=base, order=np.argsort(base + t, kind="stable"))

    def search(self, keys: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """Left insertion positions of ``base + offsets`` in `keys` (offsets must keep the order)."""
        q = self.base + offsets
        out = np.empty(len(q), dtype=np.int64)
        out[self.order] = np.searchsorted(keys, q[self.order], side="left")
        return out


class Lookup:
    """Window statistics of one event index evaluated at one set of query points."""

    def __init__(self, index: EventIndex, queries: Queries) -> None:
        self.index = index
        self.q = queries
        self.start = queries.search(index.keys, np.zeros_like(queries.t))  # entity's first event
        self.hi = queries.search(index.keys, queries.t)                    # first event at/after t

    def count_sum(self, window: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Count and sum of events in [t - window, t); window=None means all history."""
        lo = self.start if window is None else self.q.search(self.index.keys, np.maximum(self.q.t - window, 0))
        return (self.hi - lo).astype(np.float64), self.index.csum[self.hi] - self.index.csum[lo]

    def seconds_since_last(self) -> np.ndarray:
        """Seconds since the most recent event strictly before t; -1 if none."""
        last = self.index.keys[np.maximum(self.hi - 1, 0)] - self.q.base
        return np.where(self.hi > self.start, self.q.t - last, -1).astype(np.float64)

    def seconds_since_first(self) -> np.ndarray:
        """Seconds since the first event strictly before t (account age); 0 if none."""
        first = self.index.keys[np.minimum(self.start, len(self.index.keys) - 1)] - self.q.base
        return np.where(self.hi > self.start, self.q.t - first, 0).astype(np.float64)
