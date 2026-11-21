"""Per-alert explanations with SHAP (TreeSHAP on the XGBoost booster).

SHAP values are in log-odds units: positive pushes the alert score up (more suspicious),
negative pushes it down. For each alert we return the top factors by absolute contribution.
"""
from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
import shap
import xgboost as xgb

from src.features.build import feature_label

log = logging.getLogger(__name__)


class Factor(TypedDict):
    """One explanatory factor for an alert."""

    feature: str
    label: str
    value: float
    shap: float
    direction: str  # "increases risk" | "decreases risk"
    magnitude: float


class AlertExplainer:
    """Wraps a trained booster and produces SHAP values and top-k factors."""

    def __init__(self, booster: xgb.Booster, feature_names: list[str]) -> None:
        best = getattr(booster, "best_iteration", None)
        # Explain exactly the trees used for scoring (up to the early-stopping iteration).
        self.booster = booster[: best + 1] if best is not None else booster
        self.feature_names = feature_names
        self._tree: shap.TreeExplainer | None = None
        try:
            self._tree = shap.TreeExplainer(self.booster)
        except Exception as exc:  # pragma: no cover - version-compatibility fallback
            log.warning("shap.TreeExplainer failed (%s); using XGBoost's built-in TreeSHAP", exc)

    def shap_values(self, x: np.ndarray) -> np.ndarray:
        """SHAP matrix of shape (n_rows, n_features), log-odds units."""
        x = np.asarray(x, dtype=np.float32)
        if self._tree is not None:
            return np.asarray(self._tree.shap_values(x), dtype=np.float64)
        contrib = self.booster.predict(xgb.DMatrix(x, feature_names=self.feature_names), pred_contribs=True)
        return contrib[:, :-1].astype(np.float64)  # last column is the bias term

    def base_value(self) -> float:
        """Expected model output (log-odds) the SHAP values are measured from."""
        if self._tree is not None:
            return float(np.ravel(self._tree.expected_value)[0])
        dummy = np.zeros((1, len(self.feature_names)), dtype=np.float32)
        return float(self.booster.predict(xgb.DMatrix(dummy, feature_names=self.feature_names),
                                          pred_contribs=True)[0, -1])

    def top_factors(self, x: np.ndarray, k: int = 5, values: np.ndarray | None = None) -> list[list[Factor]]:
        """Top-k factors per row by |SHAP|, with readable labels, direction and magnitude."""
        sv = self.shap_values(x) if values is None else values
        x = np.asarray(x)
        order = np.argsort(-np.abs(sv), axis=1, kind="stable")[:, :k]
        out: list[list[Factor]] = []
        for i, idx in enumerate(order):
            out.append([
                Factor(feature=self.feature_names[j], label=feature_label(self.feature_names[j]),
                       value=float(x[i, j]), shap=round(float(sv[i, j]), 4),
                       direction="increases risk" if sv[i, j] > 0 else "decreases risk",
                       magnitude=round(abs(float(sv[i, j])), 4))
                for j in idx
            ])
        return out
