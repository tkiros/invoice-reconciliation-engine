"""How one document is graded, and how per-document counts become metrics.

A prediction is {"invoice": dict | None, "discrepancies": [[kind, sku], ...], "auto": bool},
or None when the system crashed on that document (graded as wrong on everything).
"""

from collections import Counter
from decimal import Decimal, InvalidOperation

# Goes into every results file. Says exactly how the expected answers were made.
LABELS = ("synthetic invoices with generated ground truth: data/generate.py (seed 2026) draws "
          "each PDF from known values and records every injected discrepancy. Nothing is "
          "hand-labeled")

HEADER_FIELDS = ("invoice_number", "po_number", "vendor", "invoice_date", "subtotal", "tax",
                 "total")
NUMBERS = {"subtotal", "tax", "total", "qty", "unit_price"}


def norm(field: str, value):
    """Compare like an accountant: numbers as Decimals (12.5 == 12.50), everything else as
    text ignoring case and repeated spaces. Dates are already ISO strings on both sides."""
    if value is None:
        return None
    if field in NUMBERS:
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return str(value)
    return " ".join(str(value).split()).casefold()


def line_keys(invoice: dict | None) -> Counter:
    """A line matches on SKU + qty + unit price. A Counter, so duplicate lines count twice."""
    if not invoice:
        return Counter()
    return Counter((norm("sku", line["sku"]), norm("qty", line["qty"]),
                    norm("unit_price", line["unit_price"])) for line in invoice["lines"])


def score(pred: dict | None, row: dict) -> dict[str, int]:
    """Counts for one document. aggregate() turns the summed counts into metrics."""
    expected = row["expected"]
    invoice = pred["invoice"] if pred else None
    true_lines, pred_lines = line_keys(expected), line_keys(invoice)
    true_found = {(d["kind"], d["sku"]) for d in row["injected"]}
    pred_found = {(kind, sku) for kind, sku in pred["discrepancies"]} if pred else set()
    auto = bool(pred and pred["auto"])
    return {
        "docs": 1,
        "fields_ok": sum(invoice is not None and norm(f, invoice[f]) == norm(f, expected[f])
                         for f in HEADER_FIELDS),
        "fields": len(HEADER_FIELDS),
        "line_tp": sum((true_lines & pred_lines).values()),
        "line_pred": sum(pred_lines.values()),
        "line_true": sum(true_lines.values()),
        "found_tp": len(true_found & pred_found),
        "found_pred": len(pred_found),
        "found_true": len(true_found),
        "auto": int(auto),
        "bad": int(bool(true_found)),
        "bad_auto": int(bool(true_found) and auto),
    }


def ratio(part: int, whole: int) -> float | None:
    """None when the denominator is 0: 'not defined' is shown as n/a, never as 0 or 1."""
    return part / whole if whole else None


def aggregate(rows: list[dict[str, int]]) -> dict[str, float | None]:
    """Corpus-level (micro) metrics: sum the counts first, then divide."""
    t = {key: sum(r[key] for r in rows) for key in rows[0]}
    return {
        "field_accuracy": ratio(t["fields_ok"], t["fields"]),
        # F1 = 2PR / (P + R), which simplifies to 2TP / (predicted + true).
        "line_f1": ratio(2 * t["line_tp"], t["line_pred"] + t["line_true"]),
        "discrepancy_precision": ratio(t["found_tp"], t["found_pred"]),
        "discrepancy_recall": ratio(t["found_tp"], t["found_true"]),
        "auto_processed": ratio(t["auto"], t["docs"]),
        "bad_auto_processed": ratio(t["bad_auto"], t["bad"]),
    }
