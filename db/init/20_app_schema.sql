-- Runs once, on a fresh volume, after the image's own 10_bootstrap_paradedb.sh
-- (which already creates vector and pg_search in this database).
-- Schema changes later: add a new numbered file here and `make reset` (dev data only).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;

CREATE TABLE IF NOT EXISTS llm_calls (
    id            bigserial PRIMARY KEY,
    project       text          NOT NULL,
    step          text          NOT NULL,      -- e.g. 'classify', 'extract'
    run_id        text,                        -- set by the eval harness; NULL otherwise
    ref           text,                        -- what the call was about (here: the PDF's sha256)
    model         text          NOT NULL,      -- model that actually served the call
    stop_reason   text,
    input_tokens  int           NOT NULL,
    output_tokens int           NOT NULL,
    latency_ms    int           NOT NULL,
    cost_usd      numeric(12,6),               -- NULL = price unknown, never a guess
    created_at    timestamptz   NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS llm_calls_run_id ON llm_calls (run_id);
