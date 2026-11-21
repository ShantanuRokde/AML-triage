"""SQLite storage for alerts, account history, narratives and analyst decisions.

`alerts` and `transactions` are rebuilt by `python -m src.pipeline build-alerts`.
`narratives` and `decisions` hold human work and are never dropped by a rebuild.
`is_laundering` / `typology` are ground truth kept for offline evaluation only; the API
must never show them to analysts.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id       INTEGER PRIMARY KEY,   -- = txn_id
    rank           INTEGER NOT NULL,
    score          REAL NOT NULL,
    ts             TEXT NOT NULL,
    src            TEXT NOT NULL,
    dst            TEXT NOT NULL,
    src_bank       TEXT,
    dst_bank       TEXT,
    amount         REAL NOT NULL,
    pay_currency   TEXT,
    recv_currency  TEXT,
    payment_format TEXT,
    features_json  TEXT NOT NULL,
    factors_json   TEXT NOT NULL,
    is_laundering  INTEGER,               -- ground truth: evaluation only, never exposed
    typology       TEXT                   -- ground truth: evaluation only, never exposed
);
CREATE INDEX IF NOT EXISTS idx_alerts_score ON alerts(score DESC);

CREATE TABLE IF NOT EXISTS transactions (
    txn_id         INTEGER PRIMARY KEY,
    ts             TEXT NOT NULL,
    src            TEXT NOT NULL,
    dst            TEXT NOT NULL,
    src_bank       TEXT,
    dst_bank       TEXT,
    amount         REAL NOT NULL,
    pay_currency   TEXT,
    recv_currency  TEXT,
    payment_format TEXT
);
CREATE INDEX IF NOT EXISTS idx_txn_src_ts ON transactions(src, ts);
CREATE INDEX IF NOT EXISTS idx_txn_dst_ts ON transactions(dst, ts);

CREATE TABLE IF NOT EXISTS narratives (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id        INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    provider        TEXT NOT NULL,
    model           TEXT,
    payload_json    TEXT NOT NULL,
    validation_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_narr_alert ON narratives(alert_id, id);

CREATE TABLE IF NOT EXISTS decisions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id       INTEGER NOT NULL,
    decision       TEXT NOT NULL CHECK (decision IN ('approve', 'dismiss')),
    note           TEXT NOT NULL DEFAULT '',
    narrative_text TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dec_alert ON decisions(alert_id, id);
"""

ALERT_COLUMNS = ["alert_id", "rank", "score", "ts", "src", "dst", "src_bank", "dst_bank", "amount",
                 "pay_currency", "recv_currency", "payment_format", "features_json", "factors_json",
                 "is_laundering", "typology"]
TXN_COLUMNS = ["txn_id", "ts", "src", "dst", "src_bank", "dst_bank", "amount", "pay_currency",
               "recv_currency", "payment_format"]


def connect(path: Path) -> sqlite3.Connection:
    """Open the database (creating folders and schema if needed) with row access by name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _replace(conn: sqlite3.Connection, table: str, df: pd.DataFrame, columns: list[str]) -> None:
    """Delete all rows of `table` and bulk-insert `df[columns]` in one transaction."""
    placeholders = ",".join("?" for _ in columns)
    rows = df[columns].itertuples(index=False, name=None)
    with conn:
        conn.execute(f"DELETE FROM {table}")
        conn.executemany(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})", rows)


def replace_alerts(conn: sqlite3.Connection, alerts: pd.DataFrame) -> None:
    """Replace the alert queue (narratives/decisions are kept)."""
    _replace(conn, "alerts", alerts, ALERT_COLUMNS)


def replace_transactions(conn: sqlite3.Connection, txns: pd.DataFrame) -> None:
    """Replace the account-history transactions."""
    _replace(conn, "transactions", txns, TXN_COLUMNS)
