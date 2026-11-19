"""Transaction-level feature engineering. Every behavioural feature uses only data strictly before t."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.features.windows import Lookup, Queries, build_index, to_seconds

log = logging.getLogger(__name__)

WINDOWS: dict[str, int] = {"1h": 3600, "24h": 86400, "7d": 7 * 86400}
# Fixed category list (union of SAML-D and HI-Small formats) so columns never depend on the data subset.
PAYMENT_FORMATS = [
    "ACH", "Cash Deposit", "Cash Withdrawal", "Cheque", "Credit card", "Cross-border", "Debit card",
    "Bitcoin", "Cash", "Credit Card", "Reinvestment", "Wire",
]

FEATURE_LABELS: dict[str, str] = {
    "log_amount": "Transaction amount (log)",
    "amount": "Transaction amount",
    "hour": "Hour of day",
    "dayofweek": "Day of week",
    "cross_currency": "Payment and received currency differ",
    "cross_border": "Sender and receiver bank differ (cross-bank/border)",
    "same_account": "Sender equals receiver",
    "is_round_100": "Round amount (multiple of 100)",
    "is_round_1000": "Round amount (multiple of 1,000)",
    "below_10k_band": "Amount just below 10,000 (9,000-9,999)",
    "src_out_count_all": "Sender: prior outgoing transactions (all time)",
    "src_out_mean_all": "Sender: mean prior outgoing amount",
    "amount_vs_src_mean": "Amount relative to sender's prior mean",
    "src_velocity_24h_vs_7d": "Sender velocity: 24h outflow vs daily 7d average",
    "src_secs_since_prev_out": "Sender: seconds since previous outgoing",
    "src_in_count_24h": "Sender: incoming transactions (24h)",
    "src_in_sum_24h": "Sender: incoming amount (24h)",
    "src_in_count_7d": "Sender: incoming transactions (7d)",
    "src_in_sum_7d": "Sender: incoming amount (7d)",
    "passthrough_ratio_24h": "Sender: incoming 24h amount / this amount (pass-through)",
    "dst_out_count_24h": "Receiver: outgoing transactions (24h)",
    "dst_in_count_all": "Receiver: prior incoming transactions (all time)",
    "dst_secs_since_prev_in": "Receiver: seconds since previous incoming",
    "pair_prev_count": "Prior transactions between this sender and receiver",
    "src_fan_out_all": "Sender: unique receivers so far (fan-out)",
    "src_new_counterparties_7d": "Sender: new receivers in last 7d",
    "dst_fan_in_all": "Receiver: unique senders so far (fan-in)",
    "dst_new_counterparties_7d": "Receiver: new senders in last 7d",
    "src_age_days": "Sender: days since first seen",
    "dst_age_days": "Receiver: days since first seen",
    "src_wcc_size_log": "Sender: connected component size (log, past graph)",
    "src_in_cycle": "Sender is on a cycle (past graph)",
    "dst_in_cycle": "Receiver is on a cycle (past graph)",
}
for _w in WINDOWS:
    FEATURE_LABELS[f"src_out_count_{_w}"] = f"Sender: outgoing transactions ({_w})"
    FEATURE_LABELS[f"src_out_sum_{_w}"] = f"Sender: outgoing amount ({_w})"
    FEATURE_LABELS[f"dst_in_count_{_w}"] = f"Receiver: incoming transactions ({_w})"
    FEATURE_LABELS[f"dst_in_sum_{_w}"] = f"Receiver: incoming amount ({_w})"
for _f in PAYMENT_FORMATS:
    FEATURE_LABELS[f"fmt_{_f}"] = f"Payment format: {_f}"


def feature_label(name: str) -> str:
    """Human-readable label for a feature name."""
    return FEATURE_LABELS.get(name, name)


def _encode_entities(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integer codes for sender, receiver and (sender, receiver) pair. Codes are ids only."""
    codes, _ = pd.factorize(pd.concat([df["src"], df["dst"]], ignore_index=True))
    n = len(df)
    src, dst = codes[:n].astype(np.int64), codes[n:].astype(np.int64)
    pair, _ = pd.factorize(src * (codes.max() + 1) + dst)
    return src, dst, pair.astype(np.int64)


def static_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Features that depend only on the transaction itself."""
    amt = df["amount"].to_numpy()
    orig = np.round(df["amount_orig"].to_numpy(), 2)
    out: dict[str, np.ndarray] = {
        "amount": amt,
        "log_amount": np.log1p(amt),
        "hour": df["ts"].dt.hour.to_numpy(),
        "dayofweek": df["ts"].dt.day_of_week.to_numpy(),
        "cross_currency": (df["pay_currency"] != df["recv_currency"]).to_numpy(),
        "cross_border": (df["src_bank"] != df["dst_bank"]).to_numpy(),
        "same_account": (df["src"] == df["dst"]).to_numpy(),
        "is_round_100": np.isclose(np.mod(orig, 100), 0),
        "is_round_1000": np.isclose(np.mod(orig, 1000), 0),
        "below_10k_band": (orig >= 9000) & (orig < 10000),
    }
    fmt = df["payment_format"].to_numpy()
    for f in PAYMENT_FORMATS:
        out[f"fmt_{f}"] = fmt == f
    return out


def behavioural_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Rolling, velocity, fan-in/out and history features computed from strictly earlier events."""
    t = to_seconds(df["ts"])
    amt = df["amount"].to_numpy()
    src, dst, pair = _encode_entities(df)
    q_src, q_dst, q_pair = Queries.make(src, t), Queries.make(dst, t), Queries.make(pair, t)
    out_idx = build_index(src, t, amt)  # outgoing events keyed by sender
    in_idx = build_index(dst, t, amt)   # incoming events keyed by receiver
    src_out, dst_in = Lookup(out_idx, q_src), Lookup(in_idx, q_dst)
    f: dict[str, np.ndarray] = {}
    for name, w in WINDOWS.items():
        f[f"src_out_count_{name}"], f[f"src_out_sum_{name}"] = src_out.count_sum(w)
        f[f"dst_in_count_{name}"], f[f"dst_in_sum_{name}"] = dst_in.count_sum(w)
    cnt_all, sum_all = src_out.count_sum()
    f["src_out_count_all"] = cnt_all
    f["src_out_mean_all"] = np.divide(sum_all, cnt_all, out=np.zeros_like(sum_all), where=cnt_all > 0)
    f["amount_vs_src_mean"] = np.divide(amt, f["src_out_mean_all"], out=np.ones_like(amt), where=cnt_all > 0)
    f["src_velocity_24h_vs_7d"] = f["src_out_sum_24h"] / (f["src_out_sum_7d"] / 7.0 + 1.0)
    f["src_secs_since_prev_out"] = src_out.seconds_since_last()
    # Money received by the sender shortly before it pays out (layering / pass-through).
    src_in = Lookup(in_idx, q_src)
    for name in ("24h", "7d"):
        f[f"src_in_count_{name}"], f[f"src_in_sum_{name}"] = src_in.count_sum(WINDOWS[name])
    f["passthrough_ratio_24h"] = np.clip(f["src_in_sum_24h"] / amt, 0, 10)
    f["dst_out_count_24h"], _ = Lookup(out_idx, q_dst).count_sum(WINDOWS["24h"])
    f["dst_in_count_all"], _ = dst_in.count_sum()
    f["dst_secs_since_prev_in"] = dst_in.seconds_since_last()
    f["pair_prev_count"], _ = Lookup(build_index(pair, t), q_pair).count_sum()
    # Unique counterparties: count first occurrences of each pair that happened before t.
    first = ~pd.Series(pair).duplicated().to_numpy()
    fo_out = Lookup(build_index(src[first], t[first]), q_src)
    fo_in = Lookup(build_index(dst[first], t[first]), q_dst)
    f["src_fan_out_all"], _ = fo_out.count_sum()
    f["src_new_counterparties_7d"], _ = fo_out.count_sum(WINDOWS["7d"])
    f["dst_fan_in_all"], _ = fo_in.count_sum()
    f["dst_new_counterparties_7d"], _ = fo_in.count_sum(WINDOWS["7d"])
    seen = build_index(np.concatenate([src, dst]), np.concatenate([t, t]))
    f["src_age_days"] = Lookup(seen, q_src).seconds_since_first() / 86400.0
    f["dst_age_days"] = Lookup(seen, q_dst).seconds_since_first() / 86400.0
    return f


def build_features(df: pd.DataFrame, use_graph: bool = False, graph_period_days: int = 30) -> pd.DataFrame:
    """Build the float32 feature matrix for a canonical, time-sorted transaction frame."""
    if not df["ts"].is_monotonic_increasing:
        raise ValueError("build_features requires a time-sorted frame")
    cols = {**static_features(df), **behavioural_features(df)}
    if use_graph:
        from src.features.graph import graph_features
        cols.update(graph_features(df, period_days=graph_period_days))
    feats = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in cols.items()}, index=df.index)
    log.info("built %d features for %s rows", feats.shape[1], f"{len(feats):,}")
    return feats
