"""End-to-end training pipeline.

    python -m src.pipeline train [--sample N] [--graph] [--synthetic]

load -> clean -> time split -> leakage-safe features -> Isolation Forest + XGBoost ->
threshold for the 1% alert budget (chosen on validation) -> metrics on val/test ->
error analysis by typology. Writes:
    models/xgb_model.json, models/iforest.joblib, models/metrics.json, models/error_analysis.json
    data/processed/scores.parquet (val + test scores for both models)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from src.config import ALERT_BUDGET, SEED, Settings, get_settings
from src.data.loader import load_transactions, to_canonical
from src.data.split import time_split
from src.data.synthetic import make_synthetic
from src.api import db
from src.eval import error_analysis
from src.explain.alerts import build_alerts, history_transactions
from src.explain.shap_explain import AlertExplainer
from src.eval.metrics import evaluate
from src.features.build import build_features
from src.models import xgb as xgb_model
from src.models.baseline import score_isolation_forest, split_summary, train_isolation_forest
from src.models.threshold import threshold_for_budget

log = logging.getLogger(__name__)


def load_dataset(settings: Settings, sample: int | None, synthetic: bool) -> tuple[pd.DataFrame, str, str]:
    """Return (canonical frame, schema, dataset label). Synthetic data is for demos/tests only."""
    if synthetic:
        df, schema = to_canonical(make_synthetic())
        return df, schema, "synthetic-demo"
    df, schema = load_transactions(settings.data_path, sample=sample)
    if schema == "hi-small" and settings.patterns_path.exists():
        df = error_analysis.attach_hismall_typology(df, error_analysis.parse_hismall_patterns(settings.patterns_path))
    return df, schema, settings.data_path.name


def split_slices(labels: pd.Series) -> dict[str, slice]:
    """Rows are time-sorted, so each split is one contiguous slice (views, no 2 GB copies)."""
    lab = labels.to_numpy()
    n_tr, n_va = int((lab == "train").sum()), int((lab == "val").sum())
    return {"train": slice(0, n_tr), "val": slice(n_tr, n_tr + n_va), "test": slice(n_tr + n_va, len(lab))}


def _model_report(y: np.ndarray, s_val: np.ndarray, s_test: np.ndarray, sl: dict[str, slice]) -> dict[str, object]:
    """Threshold from validation (alert budget) and metrics on val and test."""
    thr = threshold_for_budget(s_val, ALERT_BUDGET)
    return {"threshold": thr, "val": evaluate(y[sl["val"]], s_val, thr), "test": evaluate(y[sl["test"]], s_test, thr)}


def _print_summary(models: dict[str, dict[str, object]]) -> None:
    """Console table of headline metrics."""
    rows = []
    for name, rep in models.items():
        for split in ("val", "test"):
            m = rep[split]
            t = m["at_threshold"]
            rows.append({
                "model": name, "split": split, "PR-AUC": m["pr_auc"], "random": m["positive_rate"],
                "recall@1%": m["recall_at_budget"], "prec@1%": m["precision_at_budget"],
                "prec@100": m["precision_at_k"]["100"], "alert_rate@thr": t["alert_rate"],
                "TP": t["tp"], "FP": t["fp"], "FN": t["fn"],
            })
    print("\n" + pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def train(sample: int | None = None, use_graph: bool = False, synthetic: bool = False,
          settings: Settings | None = None) -> dict[str, object]:
    """Run the full training + evaluation pipeline and persist models, metrics and scores."""
    settings = settings or get_settings()
    t0 = time.time()
    df, schema, dataset = load_dataset(settings, sample, synthetic)
    labels, bounds = time_split(df["ts"])
    sl = split_slices(labels)
    feats = build_features(df, use_graph=use_graph)
    names = list(feats.columns)
    x = feats.to_numpy()
    del feats
    y = df["is_laundering"].to_numpy().astype(np.int8)

    log.info("training Isolation Forest baseline")
    iforest = train_isolation_forest(x[sl["train"]])
    if_val, if_test = score_isolation_forest(iforest, x[sl["val"]]), score_isolation_forest(iforest, x[sl["test"]])

    log.info("training XGBoost")
    booster = xgb_model.train_xgb(x[sl["train"]], y[sl["train"]], x[sl["val"]], y[sl["val"]], names)
    xg_val, xg_test = xgb_model.predict(booster, x[sl["val"]]), xgb_model.predict(booster, x[sl["test"]])

    models = {"xgboost": _model_report(y, xg_val, xg_test, sl),
              "isolation_forest": _model_report(y, if_val, if_test, sl)}
    models["xgboost"]["best_iteration"] = int(booster.best_iteration)
    models["xgboost"]["top_importance"] = xgb_model.top_importance(booster)
    metrics: dict[str, object] = {
        "dataset": dataset, "schema": schema, "rows": int(len(df)), "sample": sample, "use_graph": use_graph,
        "seed": SEED, "alert_budget": ALERT_BUDGET, "split_boundaries": bounds,
        "splits": json.loads(split_summary(df, labels).reset_index().to_json(orient="records")),
        "features": names, "models": models,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t0, 1),
    }

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    xgb_model.save(booster, settings.model_dir / "xgb_model.json")
    joblib.dump(iforest, settings.model_dir / "iforest.joblib")
    (settings.model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    eval_rows = slice(sl["val"].start, sl["test"].stop)
    pd.DataFrame({
        "txn_id": df["txn_id"].to_numpy()[eval_rows], "split": labels.to_numpy()[eval_rows],
        "xgboost_score": np.concatenate([xg_val, xg_test]),
        "isolation_forest_score": np.concatenate([if_val, if_test]),
        "is_laundering": y[eval_rows], "typology": df["typology"].to_numpy()[eval_rows],
    }).to_parquet(settings.processed_dir / "scores.parquet", index=False)

    _print_summary(models)
    error_analysis.run("test", settings)
    build_alert_store(settings, df=df, x=x, labels=labels, booster=booster)
    print(f"\nDone in {round(time.time() - t0, 1)}s. Metrics: {settings.model_dir / 'metrics.json'}")
    return metrics


def build_alert_store(settings: Settings | None = None, df: pd.DataFrame | None = None,
                      x: np.ndarray | None = None, labels: pd.Series | None = None,
                      booster: object | None = None) -> pd.DataFrame:
    """Score the test split, explain alerts with SHAP and write alerts + history to SQLite.

    Without arguments it reloads the data subset and features recorded in metrics.json and the
    saved model, so alerts can be rebuilt without retraining.
    """
    settings = settings or get_settings()
    metrics = json.loads((settings.model_dir / "metrics.json").read_text())
    if df is None or x is None or labels is None:
        df, _, _ = load_dataset(settings, metrics["sample"], metrics["dataset"] == "synthetic-demo")
        labels, _ = time_split(df["ts"])
        feats = build_features(df, use_graph=metrics["use_graph"])
        if list(feats.columns) != metrics["features"]:
            raise RuntimeError("rebuilt features do not match the trained model's feature list")
        x = feats.to_numpy()
        del feats
    booster = booster or xgb_model.load(settings.model_dir / "xgb_model.json")
    test = split_slices(labels)["test"]
    scores = xgb_model.predict(booster, x[test])
    explainer = AlertExplainer(booster, metrics["features"])
    alerts = build_alerts(df.iloc[test], x[test], scores, metrics["models"]["xgboost"]["threshold"], explainer)
    history = history_transactions(df, alerts)
    conn = db.connect(settings.db_path)
    db.replace_alerts(conn, alerts)
    db.replace_transactions(conn, history)
    conn.close()
    log.info("wrote %s alerts and %s history transactions to %s", f"{len(alerts):,}", f"{len(history):,}",
             settings.db_path)
    return alerts


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="AML monitoring pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="train models and evaluate")
    tr.add_argument("--sample", type=int, default=None, help="use the earliest N transactions only")
    tr.add_argument("--graph", action="store_true", help="add past-only graph snapshot features")
    tr.add_argument("--synthetic", action="store_true", help="use the tiny synthetic dataset (demo/tests only)")
    sub.add_parser("build-alerts", help="score test split with the saved model, explain alerts, write SQLite")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "train":
        train(sample=args.sample, use_graph=args.graph, synthetic=args.synthetic)
    elif args.command == "build-alerts":
        alerts = build_alert_store()
        print(f"\n{len(alerts):,} alerts written. Top 3:")
        for _, a in alerts.head(3).iterrows():
            print(f"\n#{a['rank']} alert {a['alert_id']}  score={a['score']:.4f}  {a['ts']}  "
                  f"{a['src']} -> {a['dst']}  {a['amount']:,.2f} {a['pay_currency']}  {a['payment_format']}")
            for f in json.loads(a["factors_json"]):
                print(f"   {f['shap']:+.3f}  {f['label']} = {f['value']:g}  ({f['direction']})")


if __name__ == "__main__":
    main()
