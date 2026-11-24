"""Collect the facts an alert narrative may use: transaction, SHAP factors, account history.

Everything the LLM (or template) is allowed to state comes from `AlertFacts`. The validator
later checks the narrative against exactly this object.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

from pydantic import BaseModel

TS_FORMAT = "%Y-%m-%d %H:%M:%S"


class FactorFact(BaseModel):
    """One SHAP factor in readable form."""

    feature: str
    label: str
    value: float
    shap: float
    direction: str


class AccountSummary(BaseModel):
    """Account activity up to and including the alert time."""

    account: str
    bank: str
    first_seen: str | None
    sent_count: int
    sent_amount: float
    received_count: int
    received_amount: float
    unique_counterparties: int
    last_7d_count: int
    last_7d_amount: float


class HistoryItem(BaseModel):
    """A past transaction involving the sender or receiver."""

    txn_id: int
    timestamp: str
    src: str
    dst: str
    amount: float
    payment_format: str


class AlertFacts(BaseModel):
    """All grounded inputs for one alert narrative."""

    alert_id: int
    rank: int
    score: float
    timestamp: str
    src: str
    dst: str
    src_bank: str
    dst_bank: str
    amount: float
    pay_currency: str
    recv_currency: str
    payment_format: str
    factors: list[FactorFact]
    sender: AccountSummary
    receiver: AccountSummary
    history: list[HistoryItem]

    def accounts(self) -> set[str]:
        """Every account ID present in the facts."""
        ids = {self.src, self.dst}
        for h in self.history:
            ids.update((h.src, h.dst))
        return ids


def _summary(conn: sqlite3.Connection, account: str, bank: str, ts: str) -> AccountSummary:
    """Aggregate an account's activity with ts <= alert time from the history table."""
    since = (datetime.strptime(ts, TS_FORMAT) - timedelta(days=7)).strftime(TS_FORMAT)
    r = conn.execute(
        """SELECT MIN(ts),
                  SUM(src = :a), COALESCE(SUM(CASE WHEN src = :a THEN amount END), 0),
                  SUM(dst = :a), COALESCE(SUM(CASE WHEN dst = :a THEN amount END), 0),
                  COUNT(DISTINCT CASE WHEN src = :a THEN dst ELSE src END),
                  SUM(ts >= :since), COALESCE(SUM(CASE WHEN ts >= :since THEN amount END), 0)
           FROM transactions WHERE (src = :a OR dst = :a) AND ts <= :ts""",
        {"a": account, "ts": ts, "since": since},
    ).fetchone()
    return AccountSummary(
        account=account, bank=bank, first_seen=r[0], sent_count=int(r[1] or 0), sent_amount=round(r[2], 2),
        received_count=int(r[3] or 0), received_amount=round(r[4], 2), unique_counterparties=int(r[5] or 0),
        last_7d_count=int(r[6] or 0), last_7d_amount=round(r[7], 2),
    )


def account_history(conn: sqlite3.Connection, accounts: list[str], ts: str, limit: int = 8) -> list[HistoryItem]:
    """Most recent transactions (<= alert time) involving any of the accounts, oldest first."""
    marks = ",".join("?" for _ in accounts)
    rows = conn.execute(
        f"""SELECT txn_id, ts, src, dst, amount, payment_format FROM transactions
            WHERE (src IN ({marks}) OR dst IN ({marks})) AND ts <= ?
            ORDER BY ts DESC, txn_id DESC LIMIT ?""",
        [*accounts, *accounts, ts, limit],
    ).fetchall()
    return [HistoryItem(txn_id=r[0], timestamp=r[1], src=r[2], dst=r[3], amount=round(r[4], 2),
                        payment_format=r[5]) for r in reversed(rows)]


def build_facts(conn: sqlite3.Connection, alert_id: int, history_limit: int = 8) -> AlertFacts:
    """Load one alert from SQLite and assemble its facts. Raises KeyError if it doesn't exist."""
    a = conn.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)).fetchone()
    if a is None:
        raise KeyError(alert_id)
    factors = [FactorFact(feature=f["feature"], label=f["label"], value=round(f["value"], 2),
                          shap=round(f["shap"], 3), direction=f["direction"])
               for f in json.loads(a["factors_json"])]
    return AlertFacts(
        alert_id=a["alert_id"], rank=a["rank"], score=round(a["score"], 4), timestamp=a["ts"],
        src=a["src"], dst=a["dst"], src_bank=a["src_bank"], dst_bank=a["dst_bank"],
        amount=round(a["amount"], 2), pay_currency=a["pay_currency"], recv_currency=a["recv_currency"],
        payment_format=a["payment_format"], factors=factors,
        sender=_summary(conn, a["src"], a["src_bank"], a["ts"]),
        receiver=_summary(conn, a["dst"], a["dst_bank"], a["ts"]),
        history=account_history(conn, [a["src"], a["dst"]], a["ts"], history_limit),
    )
