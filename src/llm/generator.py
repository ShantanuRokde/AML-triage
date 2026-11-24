"""Narrative generation: LLM draft -> grounding validation -> one regeneration -> template fallback.

The LLM never decides anything. It turns facts into readable text for a human analyst, and
any draft containing a number, account, date or time not present in the facts is rejected.

    python -m src.llm.generator --top 3          # narratives for the 3 highest-ranked alerts
    python -m src.llm.generator --alert-id 123   # one specific alert
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3

from pydantic import BaseModel, ValidationError

from src.config import get_settings
from src.llm.facts import AlertFacts, build_facts
from src.llm.providers import LLMProvider, get_provider
from src.llm.schema import NarrativeValidation, SARNarrative, render_text
from src.llm.template import template_narrative
from src.llm.validator import validate_narrative

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You draft SAR-style (suspicious activity report) narratives for an anti-money-laundering \
analyst at a bank. You are an assistant, not a decision maker.

Rules:
1. Use ONLY the facts in the user's JSON. Do not add any outside information.
2. Every number, account ID, date and time you write must appear in the facts, written exactly as given \
(amounts may use thousands separators, e.g. 34,180.80). Do not compute new numbers such as totals, \
differences or percentages.
3. Write timestamps exactly as given (YYYY-MM-DD HH:MM:SS).
4. Describe observable indicators neutrally ("the receiver has 22 unique senders"); never state that \
anyone is guilty or that laundering occurred.
5. recommended_action is a suggestion for the HUMAN reviewer (e.g. what to verify next). Never say a \
report was filed, an account was frozen, or the alert was approved or dismissed.
6. The model score and SHAP factors explain why the transaction was flagged; mention the strongest ones.

Return ONLY a JSON object matching this JSON schema:
{schema}"""


class NarrativeResult(BaseModel):
    """Final narrative plus provenance and validation details."""

    narrative: SARNarrative
    text: str
    provider: str
    model: str | None
    validation: NarrativeValidation


def _user_prompt(facts: AlertFacts, feedback: list[str] | None) -> str:
    """Facts JSON, plus the validator's complaints when regenerating."""
    prompt = "Facts:\n" + facts.model_dump_json(indent=2)
    if feedback:
        prompt += ("\n\nYour previous draft was rejected because:\n- " + "\n- ".join(feedback)
                   + "\nRewrite it using only values that appear in the facts.")
    return prompt


def parse_llm_json(raw: str) -> SARNarrative:
    """Parse model output (tolerating ```json fences) into the schema; raises ValueError."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        return SARNarrative.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"LLM output is not a valid narrative: {exc}") from exc


def _is_auth_error(exc: Exception) -> bool:
    """True for HTTP 401/403 from OpenAI (status_code attr) or Gemini (httpx response)."""
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return status in (401, 403)


def _template_result(facts: AlertFacts, attempts: int, llm_issues: list[str], fell_back: bool) -> NarrativeResult:
    """Template narrative, validated like any other draft."""
    narrative = template_narrative(facts)
    check = validate_narrative(narrative, facts)
    validation = NarrativeValidation(passed=check.passed, issues=check.issues, attempts=attempts,
                                     fell_back=fell_back, llm_issues=llm_issues)
    return NarrativeResult(narrative=narrative, text=render_text(narrative), provider="template",
                           model=None, validation=validation)


def generate_narrative(facts: AlertFacts, provider: LLMProvider | None = None, use_env: bool = True,
                       max_attempts: int = 2) -> NarrativeResult:
    """Draft a narrative; regenerate with feedback on validation failure, then fall back to the template.

    `provider` overrides the environment; with use_env=False and no provider the template is used.
    """
    provider = provider or (get_provider() if use_env else None)
    if provider is None:
        return _template_result(facts, attempts=0, llm_issues=[], fell_back=False)
    system = SYSTEM_PROMPT.format(schema=json.dumps(SARNarrative.model_json_schema()))
    feedback: list[str] = []
    all_issues: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            narrative = parse_llm_json(provider.complete(system, _user_prompt(facts, feedback)))
            check = validate_narrative(narrative, facts)
        except Exception as exc:  # network/API/parse errors count as a failed attempt
            log.warning("LLM attempt %d failed: %s", attempt, exc)
            feedback = [f"Output error: {str(exc)[:300]}"]
            all_issues += feedback
            if _is_auth_error(exc):  # a bad/expired key won't fix itself on retry
                break
            continue
        if check.passed:
            validation = NarrativeValidation(passed=True, attempts=attempt, llm_issues=all_issues)
            return NarrativeResult(narrative=narrative, text=render_text(narrative), provider=provider.name,
                                   model=provider.model, validation=validation)
        log.info("LLM attempt %d rejected by validator: %s", attempt, check.issues)
        feedback = check.issues
        all_issues += check.issues
    return _template_result(facts, attempts=max_attempts, llm_issues=all_issues, fell_back=True)


def main() -> None:
    """Print narratives for alerts from the SQLite store."""
    parser = argparse.ArgumentParser(description="Generate SAR narrative drafts")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--alert-id", type=int)
    group.add_argument("--top", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    conn = sqlite3.connect(get_settings().db_path)
    conn.row_factory = sqlite3.Row
    ids = ([args.alert_id] if args.alert_id is not None else
           [r[0] for r in conn.execute("SELECT alert_id FROM alerts ORDER BY rank LIMIT ?", (args.top,))])
    for alert_id in ids:
        result = generate_narrative(build_facts(conn, alert_id))
        v = result.validation
        print(f"\n{'=' * 80}\nAlert {alert_id} | provider={result.provider} model={result.model} | "
              f"validation passed={v.passed} attempts={v.attempts} fell_back={v.fell_back}")
        if v.issues or v.llm_issues:
            print("issues:", v.issues, "| rejected LLM drafts:", v.llm_issues)
        print("-" * 80 + "\n" + result.text)


if __name__ == "__main__":
    main()
