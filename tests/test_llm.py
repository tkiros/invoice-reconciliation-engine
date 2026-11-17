"""common/llm.py against a fake Messages API (real SDK, mocked HTTP). No API key, no cost."""

import json
import os
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from pydantic import BaseModel, field_validator

from common import llm


class Out(BaseModel):
    label: str

    @field_validator("label")
    @classmethod
    def known(cls, v: str) -> str:
        if v not in {"refund", "other"}:
            raise ValueError("unknown label")
        return v


def reply(text: str, stop: str = "end_turn", **extra) -> dict:
    return {"text": text, "stop": stop, **extra}


def fake_api(monkeypatch, replies: list[dict], keep_db: bool = False) -> list[dict]:
    """Point llm at a fake API that answers with `replies` in order; return captured requests."""
    seen: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        seen.append({"headers": request.headers, "body": body})
        r = replies[len(seen) - 1]
        return httpx2.Response(200, json={
            "id": f"msg_{len(seen)}", "type": "message", "role": "assistant",
            "model": body["model"], "content": [{"type": "text", "text": r["text"]}],
            "stop_reason": r["stop"], "stop_sequence": None,
            "stop_details": r.get("stop_details"),
            "usage": {"input_tokens": 1000, "output_tokens": 100},
        })

    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    fake = anthropic.Anthropic(api_key="test", http_client=http, max_retries=0)
    monkeypatch.setattr(llm, "_client", fake)
    if not keep_db:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    return seen


def test_small_tier_structured_output(monkeypatch):
    seen = fake_api(monkeypatch, [reply('{"label": "refund"}')])
    out = llm.call("refund please", step="classify", schema=Out)
    assert out == Out(label="refund")
    body = seen[0]["body"]
    assert body["model"] == llm.MODELS["small"]
    assert body["output_config"]["effort"] == "low"
    assert "fallbacks" not in body  # Haiku has no server-side fallback


def test_schema_matches_what_messages_parse_sends(monkeypatch):
    seen = fake_api(monkeypatch, [reply('{"label": "refund"}'), reply('{"label": "refund"}')])
    llm.call("x", step="s", schema=Out)
    llm.client().messages.parse(
        model="m", max_tokens=10, messages=[{"role": "user", "content": "x"}], output_format=Out
    )
    assert seen[0]["body"]["output_config"]["format"] == seen[1]["body"]["output_config"]["format"]


def test_large_tier_turns_on_refusal_fallback(monkeypatch):
    seen = fake_api(monkeypatch, [reply("hello")])
    assert llm.call("hi", step="draft", tier="large") == "hello"
    req = seen[0]
    assert req["body"]["model"] == llm.MODELS["large"]
    assert req["body"]["fallbacks"] == "default"
    assert llm.FALLBACK_BETA in req["headers"]["anthropic-beta"]
    assert req["body"]["output_config"] == {"effort": "high"}


def test_refusal_raises(monkeypatch):
    details = {"type": "refusal", "category": "cyber", "explanation": "no"}
    fake_api(monkeypatch, [reply("", stop="refusal", stop_details=details)])
    with pytest.raises(llm.LLMError, match="refused"):
        llm.call("x", step="s", schema=Out)


def test_truncation_raises(monkeypatch):
    fake_api(monkeypatch, [reply('{"lab', stop="max_tokens")])
    with pytest.raises(llm.LLMError, match="max_tokens"):
        llm.call("x", step="s", schema=Out)


def test_invalid_reply_is_retried_with_the_error(monkeypatch):
    seen = fake_api(monkeypatch, [reply('{"label": "banana"}'), reply('{"label": "other"}')])
    assert llm.call("x", step="s", schema=Out) == Out(label="other")
    retry_messages = seen[1]["body"]["messages"]
    assert [m["role"] for m in retry_messages] == ["user", "assistant", "user"]
    assert "failed validation" in retry_messages[-1]["content"]


def test_gives_up_after_schema_retries(monkeypatch):
    seen = fake_api(monkeypatch, [reply('{"label": "banana"}')] * (llm.SCHEMA_RETRIES + 1))
    with pytest.raises(llm.LLMError, match="invalid after 3 attempts"):
        llm.call("x", step="s", schema=Out)
    assert len(seen) == llm.SCHEMA_RETRIES + 1


def usage(inp: int, out: int, **kw) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=inp, output_tokens=out, **kw)


def test_cost_usd():
    assert llm.cost_usd("claude-haiku-5-5", usage(1000, 100)) == pytest.approx(0.00015)
    assert llm.cost_usd("claude-opus-5-5", usage(1000, 100)) == pytest.approx(0.006)
    assert llm.cost_usd("claude-haiku-5-5", usage(200_000, 1000)) == pytest.approx(0.1025)
    assert llm.cost_usd("some-new-model", usage(1, 1)) is None
    assert llm.cost_usd("claude-opus-5-5", usage(1, 1, cache_read_input_tokens=5)) is None
    fallback = [SimpleNamespace(type="message"), SimpleNamespace(type="fallback_message")]
    assert llm.cost_usd("claude-opus-5-5", usage(1, 1, iterations=fallback)) is None


def test_schema_retries_zero_means_one_attempt(monkeypatch):
    seen = fake_api(monkeypatch, [reply('{"label": "banana"}')])
    with pytest.raises(llm.SchemaError, match="invalid after 1 attempts"):
        llm.call("x", step="s", schema=Out, schema_retries=0)
    assert len(seen) == 1


def test_content_blocks_pass_through_unchanged(monkeypatch):
    """A PDF document block reaches the API as given, before the instruction text."""
    seen = fake_api(monkeypatch, [reply('{"label": "other"}')])
    blocks = [
        {"type": "document",
         "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}},
        {"type": "text", "text": "Extract this."},
    ]
    llm.call([{"role": "user", "content": blocks}], step="extract", schema=Out, system="sys")
    body = seen[0]["body"]
    assert body["messages"] == [{"role": "user", "content": blocks}]
    assert body["system"] == "sys"


def test_every_attempt_is_recorded_in_llm_calls(monkeypatch, db):
    run_id = f"pytest-{os.getpid()}"
    monkeypatch.setenv("EVAL_RUN", run_id)
    monkeypatch.setenv("PROJECT_NAME", "pytest")
    fake_api(monkeypatch, [reply('{"label": "banana"}'), reply('{"label": "refund"}')],
             keep_db=True)
    llm.call("x", step="classify", schema=Out, ref="doc-123")
    rows = db.execute(
        "SELECT step, ref, model, input_tokens, output_tokens, cost_usd FROM llm_calls"
        " WHERE run_id = %s ORDER BY id", (run_id,)).fetchall()
    assert len(rows) == 2  # the failed attempt cost money too
    assert rows[0][:5] == ("classify", "doc-123", llm.MODELS["small"], 1000, 100)
    assert float(rows[0][5]) == pytest.approx(0.00015)
