"""Load raw transaction CSVs into one canonical schema.

Two source schemas are supported and auto-detected from the header:

* **SAML-D** (Oztas et al., 2023): Time, Date, Sender_account, Receiver_account, Amount,
  Payment_currency, Received_currency, Sender_bank_location, Receiver_bank_location,
  Payment_type, Is_laundering, Laundering_type. `Amount` is already in a single base
  currency (GBP), so no FX conversion is applied.
* **IBM AML HI-Small**: Timestamp, From Bank, Account, To Bank, Account(.1), Amount Received,
  Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering.
  Amounts are converted to USD with rates derived from the data itself.

Canonical columns (one row per transaction, sorted by time, `txn_id` = position):
    txn_id, ts, src, dst, src_bank, dst_bank, amount, amount_orig, pay_currency,
    recv_currency, payment_format, is_laundering, typology
`typology` is the dataset's own pattern label (SAML-D only). It is used for error
analysis and must never be used as a model feature.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv

log = logging.getLogger(__name__)

SAMLD_COLUMNS = [
    "Time", "Date", "Sender_account", "Receiver_account", "Amount", "Payment_currency",
    "Received_currency", "Sender_bank_location", "Receiver_bank_location", "Payment_type",
    "Is_laundering", "Laundering_type",
]
HISMALL_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received",
    "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering",
]
CANONICAL_COLUMNS = [
    "txn_id", "ts", "src", "dst", "src_bank", "dst_bank", "amount", "amount_orig",
    "pay_currency", "recv_currency", "payment_format", "is_laundering", "typology",
]


def read_header(path: Path) -> list[str]:
    """Return the stripped column names from the first line of a CSV file."""
    with open(path, newline="", encoding="utf-8") as fh:
        return [c.strip() for c in next(csv.reader(fh))]


def detect_schema(columns: list[str]) -> str:
    """Identify the source schema from header names; raise with a helpful message otherwise."""
    cols = [c.strip() for c in columns]
    if set(SAMLD_COLUMNS).issubset(cols):
        return "saml-d"
    # pandas renames the duplicated 'Account' header to 'Account.1'; accept both spellings.
    normalized = [c if not (c == "Account" and i == 4) else "Account.1" for i, c in enumerate(cols)]
    if cols[:11] == HISMALL_COLUMNS or normalized[:11] == HISMALL_COLUMNS:
        return "hi-small"
    raise ValueError(
        "Unrecognised transaction schema. Expected SAML-D columns "
        f"{SAMLD_COLUMNS} or IBM HI-Small columns {HISMALL_COLUMNS}; got {cols}"
    )


def _samld_timestamps(raw: pd.DataFrame) -> pd.Series:
    """Combine SAML-D Date + Time. Fast path when pyarrow already parsed them (date32/time32)."""
    if "__ts" in raw:
        return raw["__ts"]
    return pd.to_datetime(raw["Date"].astype(str) + " " + raw["Time"].astype(str), format="%Y-%m-%d %H:%M:%S")


def _read_samld(path: Path) -> pd.DataFrame:
    """Read SAML-D with pyarrow, parsing Date/Time natively and combining them numerically."""
    opts = pv.ConvertOptions(column_types={
        "Sender_account": pa.string(), "Receiver_account": pa.string(),
        "Date": pa.date32(), "Time": pa.time32("s"),
    })
    table = pv.read_csv(path, convert_options=opts)
    days = table["Date"].to_numpy().astype("datetime64[s]")
    secs = table["Time"].cast(pa.int32()).to_numpy().astype("timedelta64[s]")
    raw = table.drop_columns(["Date", "Time"]).to_pandas()
    raw["__ts"] = pd.Series(days + secs, index=raw.index)
    return raw


def _from_samld(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a SAML-D frame onto the canonical columns."""
    ts = _samld_timestamps(raw)
    amount = raw["Amount"].astype("float64")
    return pd.DataFrame({
        "ts": ts,
        "src": raw["Sender_account"].astype(str),
        "dst": raw["Receiver_account"].astype(str),
        "src_bank": raw["Sender_bank_location"].astype(str),
        "dst_bank": raw["Receiver_bank_location"].astype(str),
        "amount": amount,
        "amount_orig": amount,
        "pay_currency": raw["Payment_currency"].astype(str),
        "recv_currency": raw["Received_currency"].astype(str),
        "payment_format": raw["Payment_type"].astype(str),
        "is_laundering": raw["Is_laundering"].astype("int8"),
        "typology": raw["Laundering_type"].astype(str),
    })


def derive_usd_rates(paid: pd.Series, pay_cur: pd.Series, received: pd.Series, recv_cur: pd.Series) -> dict[str, float]:
    """Estimate currency->USD rates from cross-currency rows where one side is US Dollar.

    rate[c] is the median of USD amount / c amount. Currencies without a USD counterpart
    fall back to 1.0 and are logged, so nothing is silently invented.
    """
    rates: dict[str, float] = {"US Dollar": 1.0}
    to_usd = (recv_cur == "US Dollar") & (pay_cur != "US Dollar") & (paid > 0)
    from_usd = (pay_cur == "US Dollar") & (recv_cur != "US Dollar") & (received > 0)
    r1 = (received[to_usd] / paid[to_usd]).groupby(pay_cur[to_usd]).median()
    r2 = (paid[from_usd] / received[from_usd]).groupby(recv_cur[from_usd]).median()
    for cur, rate in pd.concat([r1, r2]).groupby(level=0).median().items():
        rates[str(cur)] = float(rate)
    for cur in set(pay_cur.unique()) - set(rates):
        log.warning("No USD pair found for currency %s; using rate 1.0", cur)
        rates[str(cur)] = 1.0
    return rates


def _from_hismall(raw: pd.DataFrame) -> pd.DataFrame:
    """Map an IBM HI-Small frame onto the canonical columns (amounts converted to USD)."""
    raw = raw.copy()
    raw.columns = HISMALL_COLUMNS + list(raw.columns[11:])
    paid = raw["Amount Paid"].astype("float64")
    received = raw["Amount Received"].astype("float64")
    rates = derive_usd_rates(paid, raw["Payment Currency"], received, raw["Receiving Currency"])
    return pd.DataFrame({
        "ts": pd.to_datetime(raw["Timestamp"], format="%Y/%m/%d %H:%M"),
        "src": raw["Account"].astype(str),
        "dst": raw["Account.1"].astype(str),
        "src_bank": raw["From Bank"].astype(str),
        "dst_bank": raw["To Bank"].astype(str),
        "amount": paid * raw["Payment Currency"].map(rates).astype("float64"),
        "amount_orig": paid,
        "pay_currency": raw["Payment Currency"].astype(str),
        "recv_currency": raw["Receiving Currency"].astype(str),
        "payment_format": raw["Payment Format"].astype(str),
        "is_laundering": raw["Is Laundering"].astype("int8"),
        "typology": "unknown",
    })


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing essentials or with non-positive amounts, sort by time, assign txn_id."""
    before = len(df)
    df = df.dropna(subset=["ts", "src", "dst", "amount"])
    df = df[df["amount"] > 0]
    if len(df) < before:
        log.info("clean: dropped %d invalid rows", before - len(df))
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)  # stable: keeps file order on ties
    df.insert(0, "txn_id", np.arange(len(df), dtype=np.int64))
    return df[CANONICAL_COLUMNS]


def to_canonical(raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Detect the schema of an in-memory raw frame and return (canonical frame, schema name)."""
    schema = detect_schema(list(raw.columns))
    frame = _from_samld(raw) if schema == "saml-d" else _from_hismall(raw)
    return clean(frame), schema


def load_transactions(path: Path, sample: int | None = None) -> tuple[pd.DataFrame, str]:
    """Load a raw CSV, verify its columns, and return the canonical, time-sorted frame.

    `sample` keeps only the earliest N transactions (time-ordered subset) for fast development.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transaction file not found: {path}. Set AML_DATA_PATH in .env.")
    header = read_header(path)
    schema = detect_schema(header)
    if schema == "saml-d":
        raw = _read_samld(path)
        frame = _from_samld(raw)
    else:
        id_cols = {"From Bank": str, "Account": str, "To Bank": str, "Account.1": str}
        raw = pd.read_csv(path, engine="pyarrow", dtype=id_cols)
        frame = _from_hismall(raw)
    log.info("loaded %s rows (%s schema) from %s", f"{len(raw):,}", schema, path)
    df = clean(frame)
    if sample:
        df = df.iloc[:sample].copy()
        log.info("using time-ordered sample of the earliest %s rows", f"{len(df):,}")
    return df, schema
