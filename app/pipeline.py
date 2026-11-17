"""One PDF in, one routed document out.

  process_file()      store the PDF once per unique content, extract, check, route, audit
  analyse()           the same decisions without storing anything (shared with the eval)
  correct(), approve()  what a reviewer can do
  predict(), baseline()  the two systems `make eval` compares

Routing to review uses deterministic signals only. The model's own confidence is never
asked for, because a model that is wrong is usually wrong confidently.
"""

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import anthropic
import psycopg
import pypdfium2 as pdfium

from app import extract, oracle, store
from app.reconcile import Discrepancy, check, reconcile
from app.schema import Invoice
from common import llm

MAX_PDF_BYTES = 24 * 1024 * 1024  # the API's 32 MB request limit, less base64's third
MAX_PAGES = 600

# Why a document goes to review. Empty list = auto-approved.
REASONS = {
    "unreadable_pdf": "the file could not be opened as a PDF",
    "too_large": f"over {MAX_PDF_BYTES // 2**20} MB or {MAX_PAGES} pages",
    "schema_error": "the model's reply still failed the schema after retries",
    "model_error": "the model refused, was cut off, or the API failed after retries",
    "arithmetic_mismatch": "lines, subtotal, tax and total don't add up",
    "po_not_found": "the PO number matches no purchase order",
    "unknown_sku": "a SKU is not in the product catalog",
    "discrepancy": "price, quantity or lines differ from the purchase order",
}
LINE_KINDS = {"price_mismatch", "quantity_mismatch", "missing_line", "extra_line"}

Extractor = Callable[..., Invoice]


@dataclass
class Outcome:
    invoice: Invoice | None
    discrepancies: list[Discrepancy] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    pages: int | None = None
    error: str | None = None


def choose_extractor() -> tuple[str, Extractor]:
    """Claude normally; the ground-truth oracle when EXTRACTOR=oracle (offline demo only)."""
    if os.environ.get("EXTRACTOR", "claude") == "oracle":
        return oracle.LABEL, oracle.extract
    return extract.model_name(), extract.extract


def page_count(pdf: bytes) -> int | None:
    """Pages in the PDF, or None if it can't be opened. Costs nothing, unlike a model call."""
    try:
        return len(pdfium.PdfDocument(pdf))
    except pdfium.PdfiumError:
        return None


def review_reasons(found: list[Discrepancy]) -> list[str]:
    """Turn check results into review reasons. Same input, same answer, every time."""
    kinds = {d.kind for d in found}
    reasons = []
    if "totals_mismatch" in kinds:
        reasons.append("arithmetic_mismatch")
    if "wrong_po_number" in kinds:
        reasons.append("po_not_found")
    if "unknown_sku" in kinds:
        reasons.append("unknown_sku")
    if kinds & LINE_KINDS:
        reasons.append("discrepancy")
    return reasons


def run_checks(conn: psycopg.Connection, invoice: Invoice) -> list[Discrepancy]:
    po = store.find_po(conn, invoice.po_number)
    known = store.known_skus(conn, [line.sku for line in invoice.lines])
    return check(invoice, po, known)


def analyse(conn: psycopg.Connection, pdf: bytes, extractor: Extractor,
            ref: str | None = None) -> Outcome:
    """Extract, check and route one PDF without storing anything."""
    pages = page_count(pdf)
    if pages is None:
        return Outcome(None, reasons=["unreadable_pdf"], error="PDFium could not open the file")
    if len(pdf) > MAX_PDF_BYTES or pages > MAX_PAGES:
        return Outcome(None, reasons=["too_large"], pages=pages,
                       error=f"{len(pdf)} bytes, {pages} pages")
    try:
        invoice = extractor(pdf, ref=ref)
    except llm.SchemaError as e:
        return Outcome(None, reasons=["schema_error"], pages=pages, error=str(e))
    except (llm.LLMError, anthropic.APIError) as e:  # the SDK already retried timeouts/5xx
        return Outcome(None, reasons=["model_error"], pages=pages, error=str(e))
    found = run_checks(conn, invoice)
    return Outcome(invoice, found, review_reasons(found), pages)


def process_file(conn: psycopg.Connection, pdf: bytes, filename: str) -> tuple[int, bool]:
    """Store, extract, check and route one PDF. Returns (document id, is_new).

    The file's sha256 is the idempotency key: the same bytes uploaded again return the
    first document's id and are not processed (or paid for) twice.
    """
    sha = hashlib.sha256(pdf).hexdigest()
    label, extractor = choose_extractor()
    # ponytail: one transaction spans the model call, so a crash leaves nothing half-done
    # and a re-upload simply retries. Fine for a review queue; at high volume, commit a
    # 'processing' row first and let a sweeper retry stuck ones.
    with conn.transaction():
        doc_id = store.insert_document(conn, sha, filename, pdf)
        if doc_id is None:
            existing = store.document_id_for(conn, sha)
            store.audit(conn, existing, store.SYSTEM, "duplicate_upload",
                        new=f"same bytes uploaded again as {filename!r}; not processed")
            return existing, False
        outcome = analyse(conn, pdf, extractor, ref=sha)
        store.save_extraction(conn, doc_id, outcome.invoice, label, outcome.pages, outcome.error)
        if outcome.invoice is not None:
            store.set_status(conn, doc_id, "extracted")
        store.save_checks(conn, doc_id, outcome.discrepancies, outcome.reasons)
        store.set_status(conn, doc_id, "needs_review" if outcome.reasons else "auto_approved")
    return doc_id, True


def flatten(invoice: dict | None) -> dict[str, str]:
    """{'vendor': 'Acme', 'lines[0].qty': '5', ...}: lets a correction be diffed by field."""
    if not invoice:
        return {}
    out = {key: "" if value is None else str(value)
           for key, value in invoice.items() if key != "lines"}
    for i, line in enumerate(invoice["lines"]):
        for key, value in line.items():
            out[f"lines[{i}].{key}"] = str(value)
    return out


def correct(conn: psycopg.Connection, doc_id: int, corrected: Invoice,
            reviewer: str) -> list[tuple[str, str, str]]:
    """Save a reviewer's corrections and re-run the checks on them.

    Writes one audit row per changed field (who, when, field, old, new) and returns the
    changes. The document stays in review until someone approves it.
    """
    doc = store.get_document(conn, doc_id)
    if doc is None or doc["status"] != "needs_review":
        raise ValueError("only documents waiting for review can be corrected")
    old, new = flatten(doc["invoice"]), flatten(corrected.model_dump(mode="json"))
    fields = list(new) + [f for f in old if f not in new]
    changes = [(f, old.get(f, ""), new.get(f, "")) for f in fields if old.get(f) != new.get(f)]
    with conn.transaction():
        for f, before, after in changes:
            store.audit(conn, doc_id, reviewer, "field_corrected", f, before, after)
        store.save_invoice(conn, doc_id, corrected)
        found = run_checks(conn, corrected)
        store.save_checks(conn, doc_id, found, review_reasons(found), actor=reviewer)
    return changes


def approve(conn: psycopg.Connection, doc_id: int, reviewer: str) -> None:
    """A person accepts the document as it stands, open discrepancies included."""
    doc = store.get_document(conn, doc_id)
    if doc is None or doc["status"] != "needs_review":
        raise ValueError("only documents waiting for review can be approved")
    if doc["invoice"] is None:
        raise ValueError("enter the invoice fields before approving")
    with conn.transaction():
        store.set_status(conn, doc_id, "approved", actor=reviewer)


# --- the two systems the eval compares -----------------------------------------------------

def as_prediction(invoice: Invoice | None, found: list[Discrepancy], auto: bool) -> dict:
    return {
        "invoice": invoice.model_dump(mode="json") if invoice else None,
        "discrepancies": [[d.kind, d.sku] for d in found],
        "auto": auto,
    }


def predict(conn: psycopg.Connection, pdf: bytes) -> dict:
    """The full pipeline: schema retries, arithmetic check, reconciliation, review routing."""
    outcome = analyse(conn, pdf, extract.extract, ref=hashlib.sha256(pdf).hexdigest())
    return as_prediction(outcome.invoice, outcome.discrepancies, auto=not outcome.reasons)


def baseline(conn: psycopg.Connection, pdf: bytes) -> dict:
    """The naive version: one model call trusted as-is. No retry on a schema error (it
    raises, and the eval counts an error), no arithmetic check, no review queue: every
    document is auto-processed. It still reconciles against the PO, so the comparison
    shows what validation, retries and the arithmetic check add."""
    invoice = extract.extract(pdf, ref=hashlib.sha256(pdf).hexdigest(), schema_retries=0)
    po = store.find_po(conn, invoice.po_number)
    known = store.known_skus(conn, [line.sku for line in invoice.lines])
    return as_prediction(invoice, reconcile(invoice, po, known), auto=True)
