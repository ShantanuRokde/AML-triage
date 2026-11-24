"""Request/response models for the analyst API. Ground-truth labels are never part of any response."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.llm.facts import AccountSummary, HistoryItem
from src.llm.schema import NarrativeValidation, SARNarrative

Status = Literal["pending", "approved", "dismissed"]


class AlertSummary(BaseModel):
    """One row of the ranked alert queue."""

    alert_id: int
    rank: int
    score: float
    ts: str
    src: str
    dst: str
    amount: float
    currency: str
    payment_format: str
    status: Status


class AlertPage(BaseModel):
    """A page of the alert queue."""

    total: int
    limit: int
    offset: int
    items: list[AlertSummary]


class Factor(BaseModel):
    """SHAP factor shown as a bar in the UI."""

    feature: str
    label: str
    value: float
    shap: float
    direction: str
    magnitude: float


class DecisionOut(BaseModel):
    """A stored analyst decision."""

    id: int
    alert_id: int
    decision: Literal["approve", "dismiss"]
    note: str
    narrative_text: str | None
    created_at: str
    status: Status


class NarrativeOut(BaseModel):
    """A stored narrative draft with provenance and validation."""

    id: int
    alert_id: int
    created_at: str
    provider: str
    model: str | None
    narrative: SARNarrative
    text: str
    validation: NarrativeValidation


class AlertDetail(AlertSummary):
    """Everything the analyst needs to review one alert."""

    src_bank: str
    dst_bank: str
    pay_currency: str
    recv_currency: str
    factors: list[Factor]
    features: dict[str, float]
    sender: AccountSummary
    receiver: AccountSummary
    sender_history: list[HistoryItem]
    receiver_history: list[HistoryItem]
    narrative: NarrativeOut | None
    decision: DecisionOut | None


class NarrativeRequest(BaseModel):
    """Optional provider override for one generation."""

    provider: Literal["openai", "gemini", "template"] | None = None


class DecisionRequest(BaseModel):
    """Analyst decision. A note documenting the reason is required."""

    decision: Literal["approve", "dismiss"]
    note: str = Field(min_length=1, max_length=2000)
    narrative_text: str | None = Field(default=None, max_length=20000)
