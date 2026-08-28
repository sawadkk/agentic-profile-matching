"""Shared OpenRouter client construction and rate-limit retry helper.

Every module that calls the LLM directly via the OpenAI SDK (resume_rag.py,
agent_tools.py) points at OpenRouter's OpenAI-compatible endpoint and needs
the same retry-on-429 behavior, so it lives here once instead of being
duplicated per module. matching_agent.py talks to the same endpoint through
langchain-openai's ChatOpenAI instead of this client, but reads the same
OPENROUTER_API_KEY / MODEL env vars and OPENROUTER_BASE_URL constant.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional, TypeVar

import openai

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"

_MAX_ATTEMPTS = 3
_INITIAL_BACKOFF_SECONDS = 2.0

_T = TypeVar("_T")


def get_model() -> str:
    """The configured model id, defaulting to OpenRouter's auto-selecting free tier."""
    return os.environ.get("MODEL", DEFAULT_MODEL)


def build_openai_client() -> Optional[openai.OpenAI]:
    """Build an OpenAI SDK client pointed at OpenRouter, or None if no API key is set."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        return openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to initialize OpenRouter client: %s", exc)
        return None


def call_with_retry(fn: Callable[[], _T], label: str) -> _T:
    """Call `fn`, retrying on HTTP 429 up to _MAX_ATTEMPTS times with exponential backoff.

    Re-raises the last error if every attempt is rate limited. Any non-429
    error propagates immediately without retrying.
    """
    delay = _INITIAL_BACKOFF_SECONDS
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return fn()
        except openai.RateLimitError as exc:
            if attempt == _MAX_ATTEMPTS:
                logger.warning(
                    "%s: rate limited (HTTP 429) after %d attempts, giving up: %s",
                    label, attempt, exc,
                )
                raise
            logger.warning(
                "%s: rate limited (HTTP 429), retrying in %.0fs (attempt %d/%d)",
                label, delay, attempt, _MAX_ATTEMPTS,
            )
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")  # loop always returns or raises
