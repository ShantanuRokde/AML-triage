"""Error analysis: which laundering typologies the model misses, and where false positives come from.

SAML-D carries a `Laundering_type` per transaction (canonical column `typology`). For IBM
HI-Small, typologies come from HI-Small_Patterns.txt and are joined on transaction fields.

    python -m src.eval.error_analysis
reads data/processed/scores.parquet + models/metrics.json, writes models/error_analysis.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.config import get_settings


def recall_by_typology(df: pd.DataFrame, score_col: str, threshold: float) -> pd.DataFrame:
    """Per laundering typology: count, caught at the threshold, recall and median score."""
    pos = df[df["is_laundering"] == 1]
    out = pos.assign(caught=pos[score_col] >= threshold).groupby("typology").agg(
        n=("caught", "size"), caught=("caught", "sum"), median_score=(score_col, "median"))
    out["recall"] = out["caught"] / out["n"]
    return out.sort_values("recall")


def false_positives_by_type(df: pd.DataFrame, score_col: str, threshold: float) -> pd.DataFrame:
    """Per legitimate behaviour type: count, flagged, false-positive rate and share of all false positives."""
    neg = df[df["is_laundering"] == 0]
    out = neg.assign(flagged=neg[score_col] >= threshold).groupby("typology").agg(
        n=("flagged", "size"), flagged=("flagged", "sum"))
    out["fp_rate"] = out["flagged"] / out["n"]
    out["share_of_fps"] = out["flagged"] / max(out["flagged"].sum(), 1)
    return out.sort_values("flagged", ascending=False)


_PATTERN_HEADER = re.compile(r"BEGIN LAUNDERING ATTEMPT - ([A-Z\- ]+?)(?::|$)")


def parse_hismall_patterns(path: Path) -> pd.DataFrame:
    """Parse HI-Small_Patterns.txt into rows of (ts, src, dst, amount_orig, typology)."""
    rows: list[dict[str, object]] = []
    current = None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("BEGIN"):
            m = _PATTERN_HEADER.search(line)
            current = m.group(1).strip() if m else "UNKNOWN"
        elif line.startswith("END"):
            current = None
        elif current and line:
            f = line.split(",")
            rows.append({"ts": pd.to_datetime(f[0], format="%Y/%m/%d %H:%M"), "src": f[2], "dst": f[4],
                         "amount_orig": round(float(f[7]), 2), "typology": current})
    return pd.DataFrame(rows)


def attach_hismall_typology(df: pd.DataFrame, patterns: pd.DataFrame) -> pd.DataFrame:
    """Fill `typology` for HI-Small rows by joining on (ts, src, dst, amount paid)."""
    key = ["ts", "src", "dst", "amount_orig"]
    left = df.assign(amount_orig=df["amount_orig"].round(2)).drop(columns="typology")
    merged = left.merge(patterns.drop_duplicates(key), on=key, how="left")
    merged["typology"] = merged["typology"].fillna(
        merged["is_laundering"].map({1: "UNMATCHED_LAUNDERING", 0: "NORMAL"}))
    return merged


def _records(table: pd.DataFrame) -> list[dict[str, object]]:
    """DataFrame -> JSON-friendly list of dicts with typology as a field."""
    return json.loads(table.reset_index().to_json(orient="records"))


def run(split: str = "test") -> dict[str, object]:
    """Compute per-typology recall and false-positive tables for every scored model."""
    settings = get_settings()
    metrics = json.loads((settings.model_dir / "metrics.json").read_text())
    scores = pd.read_parquet(settings.processed_dir / "scores.parquet")
    df = scores[scores["split"] == split]
    report: dict[str, object] = {"split": split}
    for model in ("xgboost", "isolation_forest"):
        thr = metrics["models"][model]["threshold"]
        col = f"{model}_score"
        rec = recall_by_typology(df, col, thr)
        fps = false_positives_by_type(df, col, thr)
        report[model] = {"recall_by_typology": _records(rec), "false_positives_by_type": _records(fps)}
        print(f"\n{model} - recall by laundering typology ({split}, threshold from validation)")
        print(rec.to_string(float_format=lambda v: f"{v:.3f}"))
        print(f"\n{model} - false positives by legitimate type ({split})")
        print(fps.to_string(float_format=lambda v: f"{v:.4f}"))
    (settings.model_dir / "error_analysis.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
