"""Charts for the Day 2 Isolation Forest baseline, saved as PNGs to output-graphs/.

Reads only what the baseline run wrote (models/baseline_metrics.json,
data/processed/iforest_scores.parquet, data/processed/split_summary.csv), so every number
on a chart is a real model output.

    python -m src.eval.plot_baseline
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from src.config import ALERT_BUDGET, ROOT, get_settings  # noqa: E402

OUT_DIR = ROOT / "output-graphs"
# Reference palette (light mode): categorical slots 1-2, text and grid tokens.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"


def _style() -> None:
    """Thin marks, recessive axes and grid, text in ink tokens."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "lines.linewidth": 2, "legend.frameon": False,
    })


def _save(fig: plt.Figure, name: str) -> Path:
    """Save a figure into output-graphs/ and close it."""
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_split_overview(summary: pd.DataFrame) -> Path:
    """Rows and laundering rate per time split (two panels, no dual axis)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(summary))
    a.bar(x, summary["rows"] / 1e6, color=BLUE, width=0.55)
    b.bar(x, summary["rate"] * 100, color=BLUE, width=0.55)
    for ax, vals, fmt in ((a, summary["rows"] / 1e6, "{:.2f}M"), (b, summary["rate"] * 100, "{:.3f}%")):
        ax.set_xticks(x, summary["split"])
        ax.grid(axis="x", visible=False)
        for xi, v in zip(x, vals):
            ax.annotate(fmt.format(v), (xi, v), ha="center", va="bottom", xytext=(0, 3),
                        textcoords="offset points", color=INK_2)
    a.set_title("Transactions per split")
    a.set_ylabel("rows (millions)")
    b.set_title("Laundering rate per split")
    b.set_ylabel("% of transactions labelled laundering")
    fig.suptitle("Time-based split of SAML-D (train → val → test, never random)", x=0.01, ha="left",
                 color=INK_2, fontsize=10, y=1.02)
    return _save(fig, "01_split_overview.png")


def plot_pr_curves(scores: pd.DataFrame, metrics: dict[str, float]) -> Path:
    """Precision-recall curves on val and test against the random baseline (positive rate)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for split, color in (("val", BLUE), ("test", ORANGE)):
        s = scores[scores["split"] == split]
        precision, recall, _ = precision_recall_curve(s["is_laundering"], s["score"])
        step = max(1, len(recall) // 4000)
        ax.plot(recall[::step], precision[::step], color=color,
                label=f"{split}: PR-AUC {metrics[f'{split}_pr_auc']:.5f}")
        ax.axhline(metrics[f"{split}_positive_rate"], color=color, linestyle=(0, (4, 3)), linewidth=1.2,
                   label=f"{split}: random = {metrics[f'{split}_positive_rate']:.5f}")
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("recall (share of laundering caught)")
    ax.set_ylabel("precision (log scale)")
    ax.set_title("Isolation Forest: precision-recall curve")
    ax.legend(loc="lower center", fontsize=9, ncol=2)
    return _save(fig, "02_precision_recall_curve.png")


def _gains(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative share of positives caught vs share of transactions flagged (highest scores first)."""
    order = np.argsort(-s, kind="stable")
    caught = np.cumsum(y[order]) / max(y.sum(), 1)
    flagged = np.arange(1, len(y) + 1) / len(y)
    return flagged, caught


def plot_recall_vs_budget(scores: pd.DataFrame) -> Path:
    """Recall as a function of alert budget on test, with the 1% budget marked."""
    s = scores[scores["split"] == "test"]
    flagged, caught = _gains(s["is_laundering"].to_numpy(), s["score"].to_numpy())
    keep = flagged <= 0.10
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(flagged[keep] * 100, caught[keep] * 100, color=BLUE, label="Isolation Forest (test)")
    ax.plot([0, 10], [0, 10], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2, label="random flagging")
    k = int(np.ceil(ALERT_BUDGET * len(s))) - 1
    ax.axvline(ALERT_BUDGET * 100, color=MUTED, linewidth=1)
    ax.plot([ALERT_BUDGET * 100], [caught[k] * 100], "o", color=BLUE, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"1% alert budget → {caught[k] * 100:.1f}% of laundering caught\n"
                f"(random flagging would catch ~1.0%)",
                (ALERT_BUDGET * 100, caught[k] * 100), xytext=(1.6, 11), textcoords="data", color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1})
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(15, caught[keep].max() * 110))
    ax.set_xlabel("% of transactions flagged as alerts")
    ax.set_ylabel("% of laundering transactions caught (recall)")
    ax.set_title("Recall vs alert budget (test split)")
    ax.legend(loc="upper left")
    return _save(fig, "03_recall_vs_alert_budget.png")


def plot_score_distribution(scores: pd.DataFrame) -> Path:
    """Density of anomaly scores for legitimate vs laundering transactions on test."""
    s = scores[scores["split"] == "test"]
    bins = np.linspace(s["score"].quantile(0.001), s["score"].quantile(0.9999), 80)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, color, name in ((0, BLUE, "legitimate"), (1, ORANGE, "laundering")):
        vals = s.loc[s["is_laundering"] == label, "score"]
        ax.hist(vals, bins=bins, density=True, histtype="step", color=color, linewidth=2,
                label=f"{name} (n={len(vals):,})")
    ax.set_xlabel("Isolation Forest anomaly score (higher = more unusual)")
    ax.set_ylabel("density")
    ax.set_title("Score distributions overlap: laundering is not 'unusual' here")
    ax.legend()
    return _save(fig, "04_score_distribution.png")


def _budget_threshold(scores: pd.DataFrame) -> float:
    """Threshold flagging the top ALERT_BUDGET share of *validation* scores (applied to test)."""
    return float(np.quantile(scores.loc[scores["split"] == "val", "score"], 1 - ALERT_BUDGET))


def plot_recall_by_typology(scores: pd.DataFrame) -> Path:
    """Test recall per laundering typology at the 1% budget threshold chosen on validation."""
    thr = _budget_threshold(scores)
    s = scores[(scores["split"] == "test") & (scores["is_laundering"] == 1)]
    tab = s.assign(caught=s["score"] >= thr).groupby("typology")["caught"].agg(["mean", "size"])
    tab = tab.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(tab) + 1.4))
    y = np.arange(len(tab))
    ax.barh(y, tab["mean"] * 100, color=BLUE, height=0.6)
    ax.set_yticks(y, [f"{t}  (n={n})" for t, n in zip(tab.index, tab["size"])])
    alert_rate = (scores.loc[scores["split"] == "test", "score"] >= thr).mean() * 100
    ax.axvline(alert_rate, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    ax.annotate(f"random flagging ≈ {alert_rate:.1f}%", (alert_rate, -0.9), xytext=(4, 0),
                textcoords="offset points", color=INK_2, fontsize=9, va="center")
    for yi, v in zip(y, tab["mean"] * 100):
        ax.annotate(f"{v:.1f}%", (v, yi), xytext=(4, 0), textcoords="offset points", va="center",
                    color=INK_2, fontsize=8, zorder=3,
                    bbox={"boxstyle": "square,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"})
    ax.set_ylim(-1.5, len(tab) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("% of this typology's test transactions caught at the 1% budget")
    ax.set_title("Isolation Forest recall by laundering typology (test)")
    return _save(fig, "05_recall_by_typology.png")


def plot_false_positives_by_type(scores: pd.DataFrame) -> Path:
    """Alert rate among legitimate test traffic, per normal behaviour type (where false positives come from)."""
    thr = _budget_threshold(scores)
    s = scores[(scores["split"] == "test") & (scores["is_laundering"] == 0)]
    tab = s.assign(flag=s["score"] >= thr).groupby("typology")["flag"].agg(["mean", "size"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(tab) + 1.4))
    y = np.arange(len(tab))
    ax.barh(y, tab["mean"] * 100, color=BLUE, height=0.6)
    ax.set_yticks(y, [f"{t}  (n={n:,})" for t, n in zip(tab.index, tab["size"])])
    for yi, v in zip(y, tab["mean"] * 100):
        ax.annotate(f"{v:.2f}%", (v, yi), xytext=(4, 0), textcoords="offset points", va="center",
                    color=INK_2, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("% of legitimate transactions of this type flagged (false-positive rate)")
    ax.set_title("Where false positives come from (test, 1% budget)")
    return _save(fig, "06_false_positives_by_normal_type.png")


def main() -> None:
    """Render all baseline charts."""
    settings = get_settings()
    metrics = json.loads((settings.model_dir / "baseline_metrics.json").read_text())
    scores = pd.read_parquet(settings.processed_dir / "iforest_scores.parquet")
    summary = pd.read_csv(settings.processed_dir / "split_summary.csv")
    OUT_DIR.mkdir(exist_ok=True)
    _style()
    paths = [
        plot_split_overview(summary),
        plot_pr_curves(scores, metrics),
        plot_recall_vs_budget(scores),
        plot_score_distribution(scores),
        plot_recall_by_typology(scores),
        plot_false_positives_by_type(scores),
    ]
    for p in paths:
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
