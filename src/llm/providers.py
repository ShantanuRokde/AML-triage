"""LLM providers selected by environment variables.

    LLM_PROVIDER   = openai | gemini | template   (default: template)
    OPENAI_API_KEY / OPENAI_MODEL   (default model: gpt-4o-mini)
    GEMINI_API_KEY / GEMINI_MODEL   (default model: gemini-2.5-flash)

If the chosen provider has no API key, `get_provider` returns None and the caller uses the
deterministic template. Providers only return raw text; parsing and validation happen in
the generator.
"""
from __future__ import annotations

import logging
import os
from typing import Protocol

import httpx

from src.config import SEED

log = logging.getLogger(__name__)
TIMEOUT_S = 60.0


class LLMProvider(Protocol):
    """Anything that turns (system, user) prompts into a JSON string."""

    name: str
    model: str

    def complete(self, system: str, user: str) -> str:
        """Return the model's raw text response."""
        ...


class OpenAIProvider:
    """OpenAI Chat Completions with JSON output mode."""

    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key, timeout=TIMEOUT_S)

    def complete(self, system: str, user: str) -> str:
        """Call the chat completions endpoint and return the message text."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.2,
            seed=SEED,
        )
        return resp.choices[0].message.content or ""


class GeminiProvider:
    """Google Gemini generateContent REST API with JSON output."""

    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._key = api_key

    def complete(self, system: str, user: str) -> str:
        """POST the prompt and return the first candidate's text."""
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        }
        resp = httpx.post(self.URL.format(model=self.model), json=body,
                          headers={"x-goog-api-key": self._key}, timeout=TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def get_provider(name: str | None = None) -> LLMProvider | None:
    """Provider from LLM_PROVIDER (or `name`); None means use the template."""
    choice = (name or os.getenv("LLM_PROVIDER", "template")).strip().lower()
    if choice == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if key:
            return OpenAIProvider(key, os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    elif choice == "gemini":
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if key:
            return GeminiProvider(key, os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    elif choice != "template":
        log.warning("Unknown LLM_PROVIDER %r; using template", choice)
        return None
    if choice != "template":
        log.info("LLM_PROVIDER=%s but no API key set; using template", choice)
    return None
