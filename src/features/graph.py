"""Optional graph features from periodic snapshots of the *past* transaction graph (NetworkX).

Time is cut into fixed calendar periods aligned to 2000-01-01. Transactions in period p get
features from the graph of all edges in periods < p, so no edge from the same or a later
period is ever visible. Period boundaries are data-independent, which keeps results stable
when the dataset is truncated (e.g. with --sample).
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

from src.features.windows import to_seconds

log = logging.getLogger(__name__)


def _snapshot_stats(graph: nx.DiGraph) -> tuple[dict[int, int], set[int]]:
    """Weakly-connected component size per node, and nodes inside a non-trivial SCC (on a cycle)."""
    wcc = {n: len(c) for c in nx.weakly_connected_components(graph) for n in c}
    on_cycle = {n for c in nx.strongly_connected_components(graph) if len(c) > 1 for n in c}
    return wcc, on_cycle


def graph_features(df: pd.DataFrame, period_days: int = 30) -> dict[str, np.ndarray]:
    """Compute src component size (log) and src/dst cycle membership from past-only snapshots."""
    codes, _ = pd.factorize(pd.concat([df["src"], df["dst"]], ignore_index=True))
    n = len(df)
    src, dst = codes[:n], codes[n:]
    period = to_seconds(df["ts"]) // (period_days * 86400)
    wcc_size = np.ones(n)
    src_cycle = np.zeros(n)
    dst_cycle = np.zeros(n)
    graph = nx.DiGraph()
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        if graph.number_of_edges():
            wcc, on_cycle = _snapshot_stats(graph)
            wcc_size[rows] = [wcc.get(int(s), 1) for s in src[rows]]
            src_cycle[rows] = [int(s) in on_cycle for s in src[rows]]
            dst_cycle[rows] = [int(d) in on_cycle for d in dst[rows]]
        graph.add_edges_from(zip(src[rows].tolist(), dst[rows].tolist()))
        log.info("graph snapshot period %d: %d nodes, %d edges", p, graph.number_of_nodes(), graph.number_of_edges())
    return {"src_wcc_size_log": np.log1p(wcc_size), "src_in_cycle": src_cycle, "dst_in_cycle": dst_cycle}
