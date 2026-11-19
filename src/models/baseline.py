"""Isolation Forest baseline (unsupervised: labels are used only for evaluation).

CLI (Day 2 verification):
    python -m src.models.baseline [--sample N] [--graph]
Loads data, prints split statistics, builds leakage-safe features, fits Isolation Forest on the
training split and reports PR-AUC on validation and test. Results go to models/baseline_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from src.config import SEED, get_settings
from src.data.loader import load_transactions
from src.data.split import time_split
from src.features.build import build_features

log = logging.getLogger(__name__)


def train_isolation_forest(x_train: np.ndarray, max_rows: int = 200_000, seed: int = SEED) -> IsolationForest:
    """Fit Isolation Forest on (a random subsample of) the training features."""
    rng = np.random.default_rng(seed)
    if len(x_train) > max_rows:
        x_train = x_train[rng.choice(len(x_train), size=max_rows, replace=False)]
    model = IsolationForest(n_estimators=200, max_samples=256, random_state=seed, n_jobs=-1)
    return model.fit(x_train)


def score_isolation_forest(model: IsolationForest, x: np.ndarray, batch: int = 500_000) -> np.ndarray:
    """Anomaly score where higher means more suspicious (negated sklearn score_samples), in batches."""
    return np.concatenate([-model.score_samples(x[i:i + batch]) for i in range(0, len(x), batch)])


def split_summary(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Rows, positives and positive rate per split, in time order."""
    out = df.groupby(labels)["is_laundering"].agg(rows="size", positives="sum", rate="mean")
    return out.loc[["train", "val", "test"]]


def run(sample: int | None, use_graph: bool) -> dict[str, object]:
    """Load -> split -> features -> Isolation Forest -> PR-AUC on val/test. Returns the metrics dict."""
    settings = get_settings()
    t0 = time.time()
    df, schema = load_transactions(settings.data_path, sample=sample)
    labels, bounds = time_split(df["ts"])
    summary = split_summary(df, labels)
    print(f"\nSchema: {schema} | rows: {len(df):,} | load {time.time() - t0:.1f}s")
    print(f"Split boundaries: {bounds}")
    print(summary.to_string(float_format=lambda v: f"{v:.5f}"))

    t1 = time.time()
    feats = build_features(df, use_graph=use_graph)
    feat_secs = time.time() - t1
    print(f"\nFeatures: {feats.shape[1]} columns in {feat_secs:.1f}s "
          f"({feats.memory_usage().sum() / 1e9:.2f} GB float32)")

    x = feats.to_numpy()
    y = df["is_laundering"].to_numpy()
    mask = {s: (labels == s).to_numpy() for s in ("train", "val", "test")}
    model = train_isolation_forest(x[mask["train"]])
    metrics: dict[str, object] = {
        "model": "IsolationForest", "schema": schema, "rows": len(df), "sample": sample,
        "use_graph": use_graph, "feature_seconds": round(feat_secs, 1), "split_boundaries": bounds,
    }
    scored: list[pd.DataFrame] = []
    for split in ("val", "test"):
        scores = score_isolation_forest(model, x[mask[split]])
        metrics[f"{split}_pr_auc"] = float(average_precision_score(y[mask[split]], scores))
        metrics[f"{split}_positive_rate"] = float(y[mask[split]].mean())
        scored.append(pd.DataFrame({
            "txn_id": df["txn_id"].to_numpy()[mask[split]], "split": split, "score": scores,
            "is_laundering": y[mask[split]], "typology": df["typology"].to_numpy()[mask[split]],
        }))
    print(f"\nIsolation Forest PR-AUC  val: {metrics['val_pr_auc']:.5f} "
          f"(random = {metrics['val_positive_rate']:.5f})  test: {metrics['test_pr_auc']:.5f} "
          f"(random = {metrics['test_positive_rate']:.5f})")

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, settings.model_dir / "iforest.joblib")
    Path(settings.model_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    # Per-transaction scores (val + test) for plots and error analysis.
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(scored, ignore_index=True).to_parquet(settings.processed_dir / "iforest_scores.parquet", index=False)
    split_summary(df, labels).reset_index().to_csv(settings.processed_dir / "split_summary.csv", index=False)
    return metrics


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Isolation Forest baseline")
    parser.add_argument("--sample", type=int, default=None, help="use the earliest N transactions only")
    parser.add_argument("--graph", action="store_true", help="add past-only graph snapshot features")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.sample, args.graph)


if __name__ == "__main__":
    main()
