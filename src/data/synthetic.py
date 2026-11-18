"""Tiny synthetic generator in the SAML-D raw schema.

Used ONLY for unit tests and an offline UI demo. Metrics produced on this data are
meaningless for real-world performance and are labelled `synthetic-demo` everywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SEED

_NORMAL_TYPES = ["Normal_Fan_Out", "Normal_Fan_In", "Normal_Small_Fan_Out", "Normal_Group", "Normal_Periodical"]
_FORMATS = ["Credit card", "Debit card", "Cheque", "ACH", "Cross-border", "Cash Deposit", "Cash Withdrawal"]
_FORMAT_P = [0.21, 0.21, 0.21, 0.21, 0.1, 0.03, 0.03]


def _row(ts: pd.Timestamp, src: str, dst: str, amount: float, fmt: str, label: int, typ: str,
         foreign: bool = False) -> dict[str, object]:
    """Build one raw SAML-D row."""
    return {
        "Time": ts.strftime("%H:%M:%S"), "Date": ts.strftime("%Y-%m-%d"),
        "Sender_account": src, "Receiver_account": dst, "Amount": round(float(amount), 2),
        "Payment_currency": "UK pounds", "Received_currency": "Euro" if foreign else "UK pounds",
        "Sender_bank_location": "UK", "Receiver_bank_location": "Spain" if foreign else "UK",
        "Payment_type": "Cross-border" if foreign else fmt, "Is_laundering": label, "Laundering_type": typ,
    }


def make_synthetic(n_accounts: int = 400, days: int = 40, txns_per_day: int = 150,
                   motifs_per_day: int = 2, seed: int = SEED) -> pd.DataFrame:
    """Generate a raw SAML-D-shaped frame with normal traffic plus fan-out/fan-in/cycle/structuring motifs."""
    rng = np.random.default_rng(seed)
    accounts = sorted({str(x) for x in rng.integers(10**8, 10**10, size=n_accounts * 2)})[:n_accounts]
    accounts = [str(a) for a in rng.permutation(accounts)]
    peers = {a: list(rng.choice(accounts, size=6, replace=False)) for a in accounts}
    start = pd.Timestamp("2022-10-07")
    rows: list[dict[str, object]] = []
    for day in range(days):
        day_start = start + pd.Timedelta(days=day)
        for _ in range(txns_per_day):
            src = accounts[rng.integers(n_accounts)]
            ts = day_start + pd.Timedelta(seconds=int(rng.integers(6 * 3600, 22 * 3600)))
            rows.append(_row(ts, src, str(rng.choice(peers[src])), rng.lognormal(8.3, 0.9),
                             str(rng.choice(_FORMATS, p=_FORMAT_P)), 0, str(rng.choice(_NORMAL_TYPES))))
        for _ in range(motifs_per_day):
            rows.extend(_motif(rng, accounts, day_start))
    return pd.DataFrame(rows)


def _motif(rng: np.random.Generator, accounts: list[str], day_start: pd.Timestamp) -> list[dict[str, object]]:
    """Inject one laundering motif with a burst of transactions inside a few hours."""
    kind = str(rng.choice(["Fan_Out", "Fan_In", "Cycle", "Structuring"]))
    hub = accounts[rng.integers(len(accounts))]
    others = [str(a) for a in rng.choice(accounts, size=6, replace=False) if a != hub]
    t0 = day_start + pd.Timedelta(seconds=int(rng.integers(0, 6 * 3600)))
    out: list[dict[str, object]] = []
    for i, other in enumerate(others):
        ts = t0 + pd.Timedelta(minutes=int(7 * i + rng.integers(0, 5)))
        if kind == "Fan_Out":
            out.append(_row(ts, hub, other, rng.uniform(9000, 9990), "ACH", 1, kind, foreign=i % 2 == 0))
        elif kind == "Fan_In":
            out.append(_row(ts, other, hub, rng.uniform(9000, 9990), "Cash Deposit", 1, kind))
        elif kind == "Structuring":
            out.append(_row(ts, hub, others[0], rng.uniform(9500, 9999), "Cash Deposit", 1, kind))
        else:  # Cycle: hub -> o1 -> o2 -> ... -> hub
            chain = [hub, *others[:4], hub]
            if i < len(chain) - 1:
                out.append(_row(ts, chain[i], chain[i + 1], 15000 - 150 * i, "Cross-border", 1, kind, foreign=True))
    return out
