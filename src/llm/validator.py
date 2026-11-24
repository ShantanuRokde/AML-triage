"""Grounding checks for narrative drafts.

Every date, time, account ID and number in the narrative must exist in the input facts.
Text is scanned in a fixed order (dates -> times -> hex IDs -> numbers) and each match is
masked before the next pass, so the digits of a date are not re-checked as loose numbers.

Numbers match if they equal an input value or its rounding to 0-4 decimals ("34,180.80",
"34180.8" and "34,181" all match 34180.80). Small integers 0-10 are allowed as structural
words ("top 5 factors", "1% budget"). Long digit strings (7+ digits, no separators) are
treated as account IDs and must be a known account (or a known number).
A few phrases that would present the draft as a decision are also flagged.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from src.llm.facts import AlertFacts
from src.llm.schema import NarrativeValidation, SARNarrative

DATE_RE = re.compile(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b")
TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")
HEX_ID_RE = re.compile(r"\b(?=[0-9A-F]*\d)(?=[0-9A-F]*[A-F])[0-9A-F]{8,}\b")
NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
STRUCTURAL_INTS = set(range(11))
DECISION_PHRASES = [
    "has been filed", "sar was filed", "have filed", "account has been frozen", "has been frozen",
    "is guilty", "confirmed money laundering", "we have reported", "has been reported to",
    "case closed", "alert dismissed", "alert approved",
]


def _mask(text: str, span: tuple[int, int]) -> str:
    """Replace a matched span with spaces so later passes ignore it."""
    return text[: span[0]] + " " * (span[1] - span[0]) + text[span[1]:]


def _leaves(obj: object) -> Iterable[object]:
    """All scalar leaves of a nested dict/list structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _leaves(v)
    else:
        yield obj


def _rounded(value: float) -> set[float]:
    """A value and its roundings to 0-4 decimals (absolute, normalised to 6 dp)."""
    v = abs(float(value))
    return {round(round(v, d), 6) for d in range(5)} | {round(v, 6)}


class AllowedFacts:
    """Sets of dates, times, accounts and numbers extracted from AlertFacts."""

    def __init__(self, facts: AlertFacts) -> None:
        self.accounts = facts.accounts()
        self.dates: set[str] = set()
        self.times: set[str] = set()
        self.numbers: set[float] = set()
        for leaf in _leaves(facts.model_dump()):
            if isinstance(leaf, bool) or leaf is None:
                continue
            if isinstance(leaf, (int, float)):
                self.numbers |= _rounded(leaf)
            elif isinstance(leaf, str):
                self._add_text(leaf)
        self.numbers |= _rounded(facts.score * 100)  # score quoted as a percentage
        for i in STRUCTURAL_INTS:
            self.numbers.add(float(i))

    def _add_text(self, text: str) -> None:
        """Collect dates/times from a fact string, then any numbers left (e.g. '24h', '10,000')."""
        for m in DATE_RE.finditer(text):
            self.dates.add(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
            text = _mask(text, m.span())
        for m in TIME_RE.finditer(text):
            hh, mm, ss = m.group(1).zfill(2), m.group(2), m.group(3)
            self.times.add(f"{hh}:{mm}")
            if ss:
                self.times.add(f"{hh}:{mm}:{ss}")
            text = _mask(text, m.span())
        if text in self.accounts:
            return
        for m in NUM_RE.finditer(text):
            self.numbers |= _rounded(float(m.group().replace(",", "")))

    def has_number(self, token: str) -> bool:
        """True if the numeric token equals an allowed value (or a rounding of one)."""
        return round(abs(float(token.replace(",", ""))), 6) in self.numbers


def narrative_strings(n: SARNarrative) -> list[str]:
    """Every free-text field of the narrative."""
    out = [n.summary, *n.suspicious_indicators, n.recommended_action]
    for e in n.timeline:
        out += [e.timestamp, e.description]
    return out


def check_text(text: str, allowed: AllowedFacts) -> list[str]:
    """Return grounding issues found in one string."""
    issues: list[str] = []
    for m in DATE_RE.finditer(text):
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if date not in allowed.dates:
            issues.append(f"Date {m.group()} does not appear in the input")
        text = _mask(text, m.span())
    for m in TIME_RE.finditer(text):
        hh, mm, ss = m.group(1).zfill(2), m.group(2), m.group(3)
        if (f"{hh}:{mm}:{ss}" if ss else f"{hh}:{mm}") not in allowed.times:
            issues.append(f"Time {m.group()} does not appear in the input")
        text = _mask(text, m.span())
    for m in HEX_ID_RE.finditer(text):
        if m.group() not in allowed.accounts:
            issues.append(f"Account ID {m.group()} does not appear in the input")
        text = _mask(text, m.span())
    for m in NUM_RE.finditer(text):
        token = m.group()
        if token.isdigit() and len(token) >= 7:
            if token not in allowed.accounts and not allowed.has_number(token):
                issues.append(f"Account ID {token} does not appear in the input")
        elif not allowed.has_number(token):
            issues.append(f"Number {token} does not appear in the input")
    lowered = text.lower()
    issues += [f"Decision language not allowed in a draft: '{p}'" for p in DECISION_PHRASES if p in lowered]
    return issues


def check_timeline(narrative: SARNarrative, facts: AlertFacts) -> list[str]:
    """Each timeline event must match a history transaction by timestamp, and the amounts and
    accounts in its description must belong to that transaction (catches values copied onto the
    wrong event, which the global existence check cannot see)."""
    by_ts: dict[str, list] = {}
    for h in facts.history:
        by_ts.setdefault(h.timestamp, []).append(h)
    issues: list[str] = []
    for e in narrative.timeline:
        items = by_ts.get(e.timestamp.strip())
        if not items:
            issues.append(f"Timeline timestamp {e.timestamp} does not match any transaction in the history")
            continue
        accounts = {a for h in items for a in (h.src, h.dst)}
        ids = {str(h.txn_id) for h in items}
        amounts: set[float] = set().union(*(_rounded(h.amount) for h in items))
        text = e.description
        for rx in (DATE_RE, TIME_RE):
            for m in rx.finditer(text):
                text = _mask(text, m.span())
        for m in NUM_RE.finditer(text):
            token = m.group()
            if token.isdigit() and len(token) >= 7:
                if token not in accounts and token not in ids:
                    issues.append(f"Timeline event at {e.timestamp} mentions {token}, which is not part of "
                                  f"that transaction")
            elif round(abs(float(token.replace(",", ""))), 6) not in amounts | {float(i) for i in STRUCTURAL_INTS}:
                issues.append(f"Timeline event at {e.timestamp} states {token}, but that transaction's "
                              f"amount is {items[0].amount:,.2f}")
    return issues


def validate_narrative(narrative: SARNarrative, facts: AlertFacts) -> NarrativeValidation:
    """Check that every date, time, account ID and number in the narrative exists in the facts,
    and that timeline events are consistent with the transactions they describe."""
    allowed = AllowedFacts(facts)
    issues: list[str] = []
    for s in narrative_strings(narrative):
        for issue in check_text(s, allowed):
            if issue not in issues:
                issues.append(issue)
    issues += [i for i in check_timeline(narrative, facts) if i not in issues]
    return NarrativeValidation(passed=not issues, issues=issues)
