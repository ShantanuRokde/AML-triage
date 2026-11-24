"""Deterministic, template-based narrative used when no LLM key is set or an LLM draft fails validation.

It only formats values from AlertFacts, so it passes the grounding validator by construction
(covered by tests).
"""
from __future__ import annotations

from src.llm.facts import AlertFacts, FactorFact
from src.llm.schema import SARNarrative, TimelineEvent


def fmt_amount(x: float) -> str:
    """Money with thousands separators and 2 decimals."""
    return f"{x:,.2f}"


def fmt_value(x: float) -> str:
    """Feature value: integer if whole, otherwise 2 decimals."""
    return f"{int(x):,}" if float(x).is_integer() else f"{x:,.2f}"


BOOLEAN_FEATURES = {"cross_currency", "cross_border", "same_account", "is_round_100", "is_round_1000",
                    "below_10k_band", "src_in_cycle", "dst_in_cycle"}


def _indicator(f: FactorFact) -> str:
    """One SHAP factor as a readable indicator (yes/no for flag features)."""
    if f.feature.startswith("fmt_") or f.feature in BOOLEAN_FEATURES:
        value = "yes" if f.value >= 0.5 else "no"
    else:
        value = fmt_value(f.value)
    return f"{f.label}: {value} ({f.direction}, SHAP {f.shap:+.3f})"


def template_narrative(facts: AlertFacts) -> SARNarrative:
    """Build a factual SAR-style draft from the facts alone."""
    s, r = facts.sender, facts.receiver
    summary = (
        f"On {facts.timestamp}, account {facts.src} ({facts.src_bank}) sent {fmt_amount(facts.amount)} "
        f"{facts.pay_currency} to account {facts.dst} ({facts.dst_bank}) via {facts.payment_format}"
        + (f", received as {facts.recv_currency}" if facts.recv_currency != facts.pay_currency else "")
        + f". The monitoring model scored this transaction {facts.score:.4f} (alert rank {facts.rank:,}). "
        f"Up to this time the sender had sent {s.sent_count:,} and received {s.received_count:,} transactions; "
        f"the receiver had received {r.received_count:,} transactions and had dealt with "
        f"{r.unique_counterparties:,} unique counterparties (sent or received)."
    )
    increasing = [f for f in facts.factors if f.direction == "increases risk"]
    indicators = [_indicator(f) for f in (increasing or facts.factors)]
    if s.last_7d_count:
        indicators.append(f"Sender activity in the 7 days before the alert: {s.last_7d_count:,} transactions "
                          f"totalling {fmt_amount(s.last_7d_amount)}")
    if r.last_7d_count:
        indicators.append(f"Receiver activity in the 7 days before the alert: {r.last_7d_count:,} transactions "
                          f"totalling {fmt_amount(r.last_7d_amount)}")
    timeline = [
        TimelineEvent(timestamp=h.timestamp,
                      description=f"{h.src} sent {fmt_amount(h.amount)} to {h.dst} via {h.payment_format}"
                      + (" (alerted transaction)" if h.txn_id == facts.alert_id else ""))
        for h in facts.history
    ]
    action = ("For the reviewing analyst: verify the indicators above against the full account records "
              "and customer due-diligence file, then decide whether to escalate for SAR consideration or "
              "dismiss with a documented reason. No automated decision has been made.")
    return SARNarrative(summary=summary, suspicious_indicators=indicators[:10], timeline=timeline,
                        recommended_action=action)
