"""Pydantic schemas for SAR-style narrative drafts and their validation result.

The narrative is a DRAFT for a human analyst. `recommended_action` is a suggestion for the
reviewer only; the system never files, escalates or dismisses anything on its own.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

DISCLAIMER = ("Draft generated automatically for analyst review. It is not a decision: "
              "a human analyst must verify the facts and decide whether to escalate or dismiss.")


class TimelineEvent(BaseModel):
    """One dated event in the alert timeline."""

    timestamp: str = Field(description="Exact timestamp copied from the input, format YYYY-MM-DD HH:MM:SS")
    description: str = Field(description="What happened, using only accounts and amounts from the input")


class SARNarrative(BaseModel):
    """Structured SAR-style narrative draft."""

    summary: str = Field(min_length=20, description="2-4 sentence factual summary of the alerted transaction")
    suspicious_indicators: list[str] = Field(min_length=1, max_length=10,
                                             description="Observable red flags, each grounded in an input fact")
    timeline: list[TimelineEvent] = Field(max_length=20, description="Chronological events from the history")
    recommended_action: str = Field(description="Suggested next step FOR THE HUMAN REVIEWER; not a decision")


class NarrativeValidation(BaseModel):
    """Outcome of grounding checks plus how the final narrative was produced."""

    passed: bool
    issues: list[str] = Field(default_factory=list)
    attempts: int = 0
    fell_back: bool = False
    llm_issues: list[str] = Field(default_factory=list, description="Issues of rejected LLM drafts, if any")


def render_text(n: SARNarrative) -> str:
    """Plain-text rendering for the analyst's editable textarea."""
    lines = ["SUMMARY", n.summary, "", "SUSPICIOUS INDICATORS"]
    lines += [f"- {i}" for i in n.suspicious_indicators]
    lines += ["", "TIMELINE"] + [f"- {e.timestamp}: {e.description}" for e in n.timeline]
    lines += ["", "RECOMMENDED ACTION (for human review only)", n.recommended_action, "", DISCLAIMER]
    return "\n".join(lines)
