"""Shared test fixtures.

- `generated`: the generated dataset (made on first use if `make data` hasn't run).
- `db`: a clean, seeded test database `invoice_test`, separate from the dev database.
  Database tests are skipped only when Postgres is unreachable (`make up` not running).
- `fake_claude`: the real Anthropic SDK pointed at a fake HTTP API. No key, no cost.
"""

import base64
import json
import os
from pathlib import Path

import anthropic
import httpx2
import psycopg
import pytest

from app import store
from common import llm
from data import generate

ROOT = Path(__file__).resolve().parent.parent
TEST_DB = "invoice_test"
PER_TEST_TABLES = "documents, discrepancies, audit_log, llm_calls"


@pytest.fixture(scope="session")
def generated() -> Path:
    out = ROOT / "data" / "generated"
    if not (out / "erp.json").exists():
        generate.main()
    return out


@pytest.fixture(scope="session")
def test_db_url(generated) -> str:
    """Create invoice_test from db/init/*.sql and load the ERP data once per session."""
    base = os.environ.get("DATABASE_URL")
    if not base:
        pytest.skip("needs Postgres: DATABASE_URL is not set (use `make up` and `make test`)")
    try:
        admin = psycopg.connect(base, autocommit=True, connect_timeout=2)
    except psycopg.Error:
        pytest.skip("needs Postgres: run `make up` (DATABASE_URL unreachable)")
    with admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {TEST_DB}")
    url = base.rsplit("/", 1)[0] + "/" + TEST_DB
    with psycopg.connect(url, autocommit=True) as conn:
        for sql in sorted((ROOT / "db" / "init").glob("*.sql")):
            conn.execute(sql.read_text())
        store.load_erp(conn, json.loads((generated / "erp.json").read_text(encoding="utf-8")))
    return url


@pytest.fixture
def db(test_db_url, monkeypatch):
    """An open connection to the test database, with per-test tables emptied.
    DATABASE_URL points at it too, so code under test (and llm_calls logging) uses it."""
    monkeypatch.setenv("DATABASE_URL", test_db_url)
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        conn.execute(f"TRUNCATE {PER_TEST_TABLES} RESTART IDENTITY")
        yield conn


@pytest.fixture
def fake_claude(monkeypatch):
    """Install a fake Messages API. Call it with reply_for(pdf_bytes) -> text, or
    (text, stop_reason); it returns the list of request bodies the fake received.
    Every reply reports 1,000 input and 100 output tokens."""

    def install(reply_for) -> list[dict]:
        seen: list[dict] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            body = json.loads(request.content)
            seen.append(body)
            reply = reply_for(request_pdf(body))
            text, stop = reply if isinstance(reply, tuple) else (reply, "end_turn")
            return httpx2.Response(200, json={
                "id": f"msg_{len(seen)}", "type": "message", "role": "assistant",
                "model": body["model"], "content": [{"type": "text", "text": text}],
                "stop_reason": stop, "stop_sequence": None, "stop_details": None,
                "usage": {"input_tokens": 1000, "output_tokens": 100},
            })

        http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
        fake = anthropic.Anthropic(api_key="test", http_client=http, max_retries=0)
        monkeypatch.setattr(llm, "_client", fake)
        return seen

    return install


def request_pdf(body: dict) -> bytes | None:
    """The PDF inside the first message's document block, if there is one."""
    content = body["messages"][0]["content"]
    for block in content if isinstance(content, list) else []:
        if block["type"] == "document":
            return base64.standard_b64decode(block["source"]["data"])
    return None
