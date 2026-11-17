"""The ONE place this project calls a model. Swap providers by rewriting this file only.

call() maps a tier to a model, asks for structured output, validates the reply with YOUR
Pydantic schema, relies on the SDK for retries/timeouts, raises on refusal/truncation, and
writes one llm_calls row per API attempt (tokens, latency, measured cost).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal

import anthropic
import psycopg
from pydantic import BaseModel, ValidationError

from common import db

log = logging.getLogger(__name__)

Tier = Literal["small", "large"]

# Which model each tier uses is a cost/quality decision -- yours. Override in .env.
MODELS: dict[str, str] = {
    "small": os.environ.get("LLM_MODEL_SMALL", "claude-haiku-5-5"),
    "large": os.environ.get("LLM_MODEL_LARGE", "claude-opus-5-5"),
}
DEFAULT_EFFORT: dict[str, str] = {"small": "low", "large": "high"}

# USD per 1M tokens (input, output): Anthropic list prices as cached on 2026-11-03.
# Re-check https://platform.claude.com/docs/en/about-claude/pricing before quoting any cost.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),  # prompts <= 100K tokens; $0.50/$2.50 above (cost_usd)
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

# Models with the server-side refusal fallback (Haiku 5.5 has none: a refusal just raises).
FALLBACK_MODELS = {
    "claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-1", "claude-fable-5-2",
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

TIMEOUT_S = 300.0  # per attempt
MAX_RETRIES = 3  # the SDK retries 408/409/429/5xx and connection errors with backoff
SCHEMA_RETRIES = 2  # extra attempts when a reply fails your schema's validators


class LLMError(RuntimeError):
    """Refused, truncated, or never passed schema validation."""


class SchemaError(LLMError):
    """The reply still failed schema validation after every allowed attempt."""


_client: anthropic.Anthropic | None = None
_warned_no_db = False


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        # Credentials: ANTHROPIC_API_KEY, or an `ant auth login` profile.
        _client = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=MAX_RETRIES)
    return _client


def call[T: BaseModel](
    prompt: str | list[dict[str, Any]],
    *,
    step: str,
    schema: type[T] | None = None,
    tier: Tier = "small",
    system: str | None = None,
    max_tokens: int = 16000,
    effort: str | None = None,
    ref: str | None = None,
    schema_retries: int = SCHEMA_RETRIES,
) -> T | str:
    """Return a validated `schema` instance, or the reply text when schema is None.

    `prompt` is a string, or a list of messages whose content may hold blocks (e.g. a PDF).
    `step` names the call in llm_calls ('classify', 'extract', ...); `ref` says what it was
    about (e.g. a document hash) so cost and latency can be grouped per item.
    `schema_retries` extra attempts are made when a reply fails validation (0 = one shot).
    Keep business rules (totals add up, order belongs to sender) OUT of the schema and check
    them after the call: a model told why it failed will bend its answer until it passes.
    """
    model = MODELS[tier]
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    output_config: dict[str, Any] = {"effort": effort or DEFAULT_EFFORT[tier]}
    if schema is not None:
        # Same strict JSON schema messages.parse() would send. create() is used instead so a
        # reply that fails validation is still logged (parse() raises and drops the usage).
        output_config["format"] = {
            "type": "json_schema",
            "schema": anthropic.transform_schema(schema),
        }
    extra: dict[str, Any] = {}
    if system:
        extra["system"] = system
    if model in FALLBACK_MODELS:
        extra |= {"betas": [FALLBACK_BETA], "fallbacks": "default"}

    for attempt in range(schema_retries + 1):
        t0 = time.perf_counter()
        resp = client().beta.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            output_config=output_config,
            **extra,
        )
        _record(step, ref, resp, int((time.perf_counter() - t0) * 1000))

        if resp.stop_reason == "refusal":
            category = resp.stop_details.category if resp.stop_details else None
            raise LLMError(f"{step}: refused (category={category})")
        if resp.stop_reason == "max_tokens":
            raise LLMError(f"{step}: hit max_tokens={max_tokens}; raise it")

        text = "".join(b.text for b in resp.content if b.type == "text")
        if schema is None:
            return text
        try:
            return schema.model_validate_json(text)
        except ValidationError as e:
            if attempt == schema_retries:
                raise SchemaError(f"{step}: invalid after {attempt + 1} attempts: {e}") from e
            messages = [
                *messages,
                {"role": "assistant", "content": resp.content},  # echoed back unchanged
                {"role": "user", "content": f"That JSON failed validation:\n{e}\n"
                                            "Return corrected JSON only."},
            ]
    raise AssertionError("unreachable")


def cost_usd(model: str, usage: Any) -> float | None:
    """Measured cost of one call, or None when it can't be priced honestly."""
    price = PRICES.get(model)
    cached = (getattr(usage, "cache_read_input_tokens", 0) or 0) + (
        getattr(usage, "cache_creation_input_tokens", 0) or 0
    )
    fallback_ran = any(
        getattr(i, "type", None) == "fallback_message"
        for i in (getattr(usage, "iterations", None) or [])
    )
    if price is None or cached or fallback_ran:
        # Unknown model; prompt caching on (add cache prices first); or billed across two models.
        return None
    inp, out = price
    if model == "claude-haiku-5-5" and usage.input_tokens > 100_000:
        inp, out = 0.50, 2.50
    return (usage.input_tokens * inp + usage.output_tokens * out) / 1_000_000


def _record(step: str, ref: str | None, resp: Any, latency_ms: int) -> None:
    global _warned_no_db
    if not os.environ.get("DATABASE_URL"):
        if not _warned_no_db:
            log.warning("DATABASE_URL unset: llm_calls rows are not being recorded")
            _warned_no_db = True
        return
    u = resp.usage
    row = (
        os.environ.get("PROJECT_NAME", "unknown"), step, os.environ.get("EVAL_RUN"), ref,
        resp.model, resp.stop_reason, u.input_tokens, u.output_tokens, latency_ms,
        cost_usd(resp.model, u),
    )
    try:
        with db.connect() as conn:
            conn.execute(
                "INSERT INTO llm_calls (project, step, run_id, ref, model, stop_reason,"
                " input_tokens, output_tokens, latency_ms, cost_usd)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                row,
            )
    except psycopg.Error:
        log.exception("could not record llm_calls row")  # the paid-for answer still returns
