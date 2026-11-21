"""Class-weighted XGBoost classifier with early stopping on validation PR-AUC."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.config import SEED

log = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, object] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_bin": 256,
    "seed": SEED,
    "nthread": -1,
}


def class_weight(y: np.ndarray) -> float:
    """scale_pos_weight = negatives / positives, so both classes carry equal total weight."""
    pos = float(np.sum(y))
    if pos == 0:
        raise ValueError("training data has no positive labels")
    return (len(y) - pos) / pos


def train_xgb(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
              feature_names: list[str], num_rounds: int = 1000, early_stopping: int = 50,
              params: dict[str, object] | None = None) -> xgb.Booster:
    """Train on the training split; stop when validation PR-AUC hasn't improved for `early_stopping` rounds."""
    p = {**DEFAULT_PARAMS, **(params or {}), "scale_pos_weight": class_weight(y_train)}
    log.info("xgboost: scale_pos_weight=%.1f, %s train rows", p["scale_pos_weight"], f"{len(y_train):,}")
    dtrain = xgb.QuantileDMatrix(x_train, label=y_train, feature_names=feature_names)
    dval = xgb.QuantileDMatrix(x_val, label=y_val, feature_names=feature_names, ref=dtrain)
    return xgb.train(
        p, dtrain, num_boost_round=num_rounds, evals=[(dval, "val")],
        early_stopping_rounds=early_stopping, verbose_eval=50,
    )


def predict(booster: xgb.Booster, x: np.ndarray) -> np.ndarray:
    """Probability scores using the trees up to the best early-stopping iteration."""
    best = getattr(booster, "best_iteration", None)
    rng = (0, best + 1) if best is not None else (0, 0)
    return booster.inplace_predict(x, iteration_range=rng)


def top_importance(booster: xgb.Booster, k: int = 15) -> list[dict[str, float | str]]:
    """Top-k features by total gain (global view; per-alert reasons come from SHAP on Day 4)."""
    gain = booster.get_score(importance_type="total_gain")
    total = sum(gain.values()) or 1.0
    ranked = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{"feature": f, "gain_share": round(g / total, 4)} for f, g in ranked]


def save(booster: xgb.Booster, path: Path) -> None:
    """Save the booster as JSON (portable across XGBoost versions)."""
    booster.save_model(str(path))


def load(path: Path) -> xgb.Booster:
    """Load a booster saved with `save`."""
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster
