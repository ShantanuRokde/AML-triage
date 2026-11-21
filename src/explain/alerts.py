"""Turn scored test transactions into an explained alert queue and account-history table."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.explain.shap_explain import AlertExplainer

log = logging.getLogger(__name__)

TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def build_alerts(txns: pd.DataFrame, x: np.ndarray, scores: np.ndarray, threshold: float,
                 explainer: AlertExplainer, k: int = 5) -> pd.DataFrame:
    """Alerts = transactions scoring at/above the threshold, ranked, with features and top-k SHAP factors.

    `txns`, `x` and `scores` are row-aligned (typically the test split).
    """
    idx = np.flatnonzero(scores >= threshold)
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    xa = x[idx]
    log.info("explaining %s alerts with SHAP", f"{len(idx):,}")
    factors = explainer.top_factors(xa, k=k)
    names = explainer.feature_names
    sub = txns.iloc[idx]
    return pd.DataFrame({
        "alert_id": sub["txn_id"].to_numpy(),
        "rank": np.arange(1, len(idx) + 1),
        "score": scores[idx].astype(float),
        "ts": sub["ts"].dt.strftime(TS_FORMAT).to_numpy(),
        "src": sub["src"].to_numpy(), "dst": sub["dst"].to_numpy(),
        "src_bank": sub["src_bank"].to_numpy(), "dst_bank": sub["dst_bank"].to_numpy(),
        "amount": sub["amount"].to_numpy().astype(float),
        "pay_currency": sub["pay_currency"].to_numpy(), "recv_currency": sub["recv_currency"].to_numpy(),
        "payment_format": sub["payment_format"].to_numpy(),
        "features_json": [json.dumps({n: round(float(v), 4) for n, v in zip(names, row)}) for row in xa],
        "factors_json": [json.dumps(f) for f in factors],
        "is_laundering": sub["is_laundering"].to_numpy().astype(int),
        "typology": sub["typology"].to_numpy(),
    })


def history_transactions(df: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    """All transactions touching any alerted account up to the last alert time (for history panels)."""
    accounts = pd.unique(np.concatenate([alerts["src"].to_numpy(), alerts["dst"].to_numpy()]))
    last_ts = pd.Timestamp(alerts["ts"].max())
    mask = (df["src"].isin(accounts) | df["dst"].isin(accounts)) & (df["ts"] <= last_ts)
    sub = df.loc[mask]
    out = sub[["txn_id", "src", "dst", "src_bank", "dst_bank", "amount", "pay_currency",
               "recv_currency", "payment_format"]].copy()
    out.insert(1, "ts", sub["ts"].dt.strftime(TS_FORMAT))
    return out
