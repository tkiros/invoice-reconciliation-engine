"""Eval harness.  python -m evals.run --split dev [--limit N] [--systems pipeline,baseline]

Runs each system in app/pipeline.py over the invoices in evals/sets/<split>.jsonl, grades
them with evals/score.py and writes evals/results-<split>.md: accuracy for all, clean and
scanned invoices, plus cost and latency measured from llm_calls. Needs `make up`, `make data`,
`make seed` and an API key. Tune on dev; run test once, when you've stopped changing things.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from app import extract, pipeline
from common import db, llm
from evals import score as scoring

HERE = Path(__file__).parent
ROOT = HERE.parent
SYSTEMS = {"pipeline": pipeline.predict, "baseline": pipeline.baseline}
GROUPS = {
    "all": lambda row: True,
    "clean": lambda row: not row["scanned"],
    "scanned": lambda row: row["scanned"],
}
METRICS = [  # (key from score.aggregate, column title)
    ("field_accuracy", "Field accuracy"),
    ("line_f1", "Line-item F1"),
    ("discrepancy_precision", "Discrepancy precision"),
    ("discrepancy_recall", "Discrepancy recall"),
    ("auto_processed", "Auto-processed"),
    ("bad_auto_processed", "Bad invoices auto-processed"),
]


def load(split: str, limit: int | None) -> list[dict]:
    """Ground-truth rows, each with its PDF bytes and their sha256 attached."""
    lines = (HERE / "sets" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if not rows:
        raise SystemExit(f"evals/sets/{split}.jsonl is empty")
    rows = rows[:limit] if limit else rows
    for row in rows:
        path = ROOT / row["file"]
        if not path.exists():
            raise SystemExit(f"{row['file']} is missing: run `make data` first")
        row["pdf"] = path.read_bytes()
        row["sha256"] = hashlib.sha256(row["pdf"]).hexdigest()
    return rows


def pct(values: list[float], p: float) -> float:
    """Nearest-rank percentile."""
    s = sorted(values)
    return s[max(0, math.ceil(p * len(s)) - 1)]


def run_system(fn, rows: list[dict], conn) -> tuple[list[dict], list[bool]]:
    """Grade every row. Returns per-row scores and, per row, whether the system crashed."""
    scores, crashed = [], []
    for row in rows:
        try:
            pred = fn(conn, row["pdf"])
        except Exception:  # a crashed case is a failed case, not a crashed eval
            pred = None
            print(f"[{row['id']}] error:\n{traceback.format_exc()}", file=sys.stderr)
        scores.append(scoring.score(pred, row))
        crashed.append(pred is None)
    return scores, crashed


def llm_by_document(conn, run_id: str) -> dict[str, tuple]:
    """sha256 -> (calls, cost or None, unpriced calls, total latency ms) for one run."""
    rows = conn.execute(
        "SELECT ref, count(*), sum(cost_usd), count(*) FILTER (WHERE cost_usd IS NULL),"
        " sum(latency_ms) FROM llm_calls WHERE run_id = %s GROUP BY ref", (run_id,)).fetchall()
    return {ref: rest for ref, *rest in rows}


def cost_and_latency(rows: list[dict], usage: dict[str, tuple]) -> dict:
    """Measured cost per document and per page; p50/p95 of each document's model time."""
    used = [usage[row["sha256"]] for row in rows if row["sha256"] in usage]
    cost = sum(float(u[1]) for u in used if u[1] is not None)
    seconds = [u[3] / 1000 for u in used]
    return {
        "calls": sum(u[0] for u in used),
        "unpriced": sum(u[2] for u in used),
        "cost_per_doc": cost / len(rows),
        "cost_per_page": cost / sum(row["pages"] for row in rows),
        "p50": pct(seconds, 0.5) if seconds else None,
        "p95": pct(seconds, 0.95) if seconds else None,
    }


def fmt(value: float | None, pattern: str = "{:.3f}") -> str:
    return "n/a" if value is None else pattern.format(value)


def has_credentials() -> bool:
    """ANTHROPIC_API_KEY, an auth token, or an `ant auth login` profile."""
    c = llm.client()
    return bool(c.api_key or c.auth_token or c.credentials)


def evaluate(names: list[str], rows: list[dict], conn, stamp: datetime, split: str) -> list:
    results = []
    for name in names:
        run_id = f"{name}-{split}-{stamp:%Y%m%dT%H%M%SZ}"
        os.environ["EVAL_RUN"] = run_id  # common/llm.py tags llm_calls rows with it
        try:
            scores, crashed = run_system(SYSTEMS[name], rows, conn)
        finally:
            os.environ.pop("EVAL_RUN", None)
        results.append((name, scores, crashed, llm_by_document(conn, run_id)))
    return results


def report(results: list, rows: list[dict], split: str, limit: int | None,
           stamp: datetime) -> str:
    scanned = sum(row["scanned"] for row in rows)
    lines = [
        f"# Eval results: {split}",
        "",
        f"Generated {stamp:%Y-%m-%d %H:%M} UTC by `python -m evals.run --split {split}`."
        " Do not edit by hand.",
        "",
        f"- Documents: {len(rows)}" + (f", the first {limit} only" if limit else "")
        + f" ({len(rows) - scanned} clean, {scanned} scanned; "
          f"{sum(row['pages'] for row in rows)} pages)",
        f"- Injected discrepancies: {sum(len(row['injected']) for row in rows)}, in "
        f"{sum(bool(row['injected']) for row in rows)} documents",
        f"- Labels: {scoring.LABELS}",
        f"- Extraction model: `{extract.model_name()}` (tier `{extract.TIER}`)",
        "",
        "## Accuracy",
        "",
        "| System | Documents | " + " | ".join(title for _, title in METRICS) + " | Errors |",
        "|---" * (len(METRICS) + 3) + "|",
    ]
    for name, scores, crashed, _ in results:
        for group, keep in GROUPS.items():
            picked = [i for i, row in enumerate(rows) if keep(row)]
            if not picked:
                continue
            metrics = scoring.aggregate([scores[i] for i in picked])
            cells = [fmt(metrics[key]) for key, _ in METRICS]
            errors = sum(crashed[i] for i in picked)
            lines.append(f"| {name} | {group} ({len(picked)}) | " + " | ".join(cells)
                         + f" | {errors} |")
    lines += ["", "## Cost and latency (measured from llm_calls)", "",
              "| System | Documents | Model calls | Cost / document | Cost / page "
              "| p50 model time / document | p95 |", "|---" * 7 + "|"]
    for name, _, _, usage in results:
        for group, keep in GROUPS.items():
            picked = [row for row in rows if keep(row)]
            if not picked:
                continue
            c = cost_and_latency(picked, usage)
            unpriced = f" (+{c['unpriced']} unpriced calls)" if c["unpriced"] else ""
            lines.append(f"| {name} | {group} ({len(picked)}) | {c['calls']} | "
                         f"${c['cost_per_doc']:.6f}{unpriced} | ${c['cost_per_page']:.6f} | "
                         f"{fmt(c['p50'], '{:.1f} s')} | {fmt(c['p95'], '{:.1f} s')} |")
    lines += ["", "Metric definitions are in the README's Eval section."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--limit", type=int, help="first N cases only (smoke run)")
    ap.add_argument("--systems", default="pipeline,baseline")
    ap.add_argument("--out", type=Path, help="default: evals/results-<split>.md")
    args = ap.parse_args(argv)

    if not os.environ.get("DATABASE_URL"):
        raise SystemExit("DATABASE_URL is not set: run `make up` and use `make eval`")
    if not has_credentials():
        raise SystemExit("No Anthropic credentials: set ANTHROPIC_API_KEY in .env (or run "
                         "`ant auth login`). Without a key, `make demo` runs offline.")
    rows = load(args.split, args.limit)
    stamp = datetime.now(UTC)
    names = args.systems.split(",")
    try:
        with db.connect() as conn:
            results = evaluate(names, rows, conn, stamp, args.split)
    except psycopg.OperationalError as e:
        raise SystemExit(f"database unreachable ({e}); run `make up` and `make seed`") from e

    out = args.out or HERE / f"results-{args.split}.md"
    out.write_text(report(results, rows, args.split, args.limit, stamp))
    print(out.read_text())
    # The pipeline routes every failure to review, so a pipeline crash is a bug. A baseline
    # crash (no retry on a schema error) is a measured outcome, shown in the Errors column.
    return 1 if any(any(crashed) for name, _, crashed, _ in results if name == "pipeline") else 0


if __name__ == "__main__":
    raise SystemExit(main())
