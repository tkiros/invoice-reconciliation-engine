"""OFFLINE DEMO ONLY: an "extractor" that returns the generator's ground truth.

It lets `make demo` show reconciliation, the review queue and the audit trail without an
API key. It is NOT a model, and nothing it produces says anything about model accuracy.

To give the review path something to catch, it misreads some scanned invoices on purpose,
always the same ones:
  sku   the first line's SKU loses its dash (AB-1234 -> AB1234): unknown SKU, missing line
  qty   the first line's quantity is misread: arithmetic and quantity mismatches
  date  the invoice date is one day off: breaks no rule, so it is auto-approved anyway
"""

import copy
import hashlib
import json
from datetime import date, timedelta
from functools import cache
from pathlib import Path

from app.schema import Invoice
from common import llm

LABEL = "ORACLE (ground truth + deliberate misreads; not a model)"
ROOT = Path(__file__).resolve().parent.parent
MISREAD_CYCLE = ["sku", "qty", "date", None]  # dealt out to scanned invoices in id order
LOOKALIKE = {"0": "8", "1": "7", "2": "7", "3": "8", "4": "9",
             "5": "6", "6": "5", "7": "1", "8": "3", "9": "4"}


@cache
def truth_rows() -> list[dict]:
    rows = []
    for split in ("dev", "test"):
        text = (ROOT / "evals" / "sets" / f"{split}.jsonl").read_text(encoding="utf-8")
        rows += [json.loads(line) for line in text.splitlines() if line.strip()]
    return rows


@cache
def truth_by_hash() -> dict[str, dict]:
    """Ground truth keyed by the sha256 of the generated PDF on this machine."""
    return {hashlib.sha256((ROOT / row["file"]).read_bytes()).hexdigest(): row
            for row in truth_rows() if (ROOT / row["file"]).exists()}


@cache
def misread_for() -> dict[str, str | None]:
    """invoice id -> which deliberate misread the oracle applies (None = read perfectly)."""
    scanned = sorted(row["id"] for row in truth_rows() if row["scanned"])
    return {doc_id: MISREAD_CYCLE[i % len(MISREAD_CYCLE)] for i, doc_id in enumerate(scanned)}


def extract(pdf: bytes, *, ref: str | None = None, **_ignored) -> Invoice:
    """Same signature as app.extract.extract, so the pipeline can't tell them apart."""
    row = truth_by_hash().get(hashlib.sha256(pdf).hexdigest())
    if row is None:
        raise llm.LLMError("oracle mode only knows the generated invoices (run `make data`)")
    data = copy.deepcopy(row["expected"])
    apply_misread(data, misread_for().get(row["id"]))
    return Invoice.model_validate(data)


def apply_misread(data: dict, kind: str | None) -> None:
    first = data["lines"][0]
    if kind == "sku":
        first["sku"] = first["sku"].replace("-", "")
    elif kind == "qty":
        qty = first["qty"]
        first["qty"] = qty[:-1] + LOOKALIKE[qty[-1]]
    elif kind == "date":
        data["invoice_date"] = (date.fromisoformat(data["invoice_date"])
                                - timedelta(days=1)).isoformat()
