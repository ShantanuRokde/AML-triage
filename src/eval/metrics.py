"""Evaluation metrics for rare-event alerting.

Headline metrics are PR-AUC and recall at the alert budget. Accuracy is meaningless at a
~0.1% positive rate (flagging nothing scores 99.9%), and ROC-AUC looks flattering because
it rewards ranking the millions of easy negatives, so neither is reported as a headline.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score

from src.config import ALERT_BUDGET


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores (stable tie-breaking by position)."""
    return np.argsort(-scores, kind="stable")[:k]


def pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision)."""
    return float(average_precision_score(y, scores))


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of laundering among the k highest-scored transactions."""
    k = min(k, len(y))
    return float(y[_top_k(scores, k)].mean()) if k else 0.0


def recall_at_budget(y: np.ndarray, scores: np.ndarray, budget: float = ALERT_BUDGET) -> float:
    """Share of all laundering caught when the top `budget` fraction of transactions are alerted."""
    k = math.ceil(budget * len(y))
    pos = y.sum()
    return float(y[_top_k(scores, k)].sum() / pos) if pos else 0.0


def confusion_at_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Confusion matrix, precision, recall and realised alert rate for a fixed score threshold."""
    flag = scores >= threshold
    tp = int(np.sum(flag & (y == 1)))
    fp = int(np.sum(flag & (y == 0)))
    fn = int(np.sum(~flag & (y == 1)))
    tn = int(np.sum(~flag & (y == 0)))
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "alerts": tp + fp, "alert_rate": (tp + fp) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def evaluate(y: np.ndarray, scores: np.ndarray, threshold: float, budget: float = ALERT_BUDGET,
             ks: tuple[int, ...] = (100, 500, 1000)) -> dict[str, object]:
    """All headline metrics for one model on one split."""
    y = np.asarray(y).astype(np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    k_budget = math.ceil(budget * len(y))
    return {
        "rows": int(len(y)), "positives": int(y.sum()), "positive_rate": float(y.mean()),
        "pr_auc": pr_auc(y, scores),
        "recall_at_budget": recall_at_budget(y, scores, budget),
        "precision_at_budget": precision_at_k(y, scores, k_budget),
        "precision_at_k": {str(k): precision_at_k(y, scores, k) for k in ks},
        "at_threshold": confusion_at_threshold(y, scores, threshold),
    }
