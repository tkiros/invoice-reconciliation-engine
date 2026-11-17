"""Compare an extracted invoice with its purchase order. Plain Python: no model involved.

The model only reads the PDF. Whether the invoice matches the PO is decided here, by code
that gives the same answer every time and can be unit-tested.
"""

from dataclasses import dataclass
from decimal import Decimal

from app.schema import TOL, Invoice, arithmetic_issues


@dataclass(frozen=True)
class POLine:
    sku: str
    qty: Decimal
    unit_price: Decimal


@dataclass(frozen=True)
class PurchaseOrder:
    po_number: str
    vendor: str
    lines: dict[str, POLine]  # keyed by SKU


@dataclass(frozen=True)
class Discrepancy:
    kind: str  # price_mismatch, quantity_mismatch, missing_line, extra_line, unknown_sku,
    #            wrong_po_number or totals_mismatch
    sku: str | None  # None for document-level kinds
    detail: str


def reconcile(inv: Invoice, po: PurchaseOrder | None, known_skus: set[str]) -> list[Discrepancy]:
    """Match invoice lines to PO lines by SKU, then compare quantity and price.

    `po` is None when the invoice's PO number matches no purchase order.
    `known_skus` is the product catalog: a SKU outside it is usually a misread.
    """
    found = [Discrepancy("unknown_sku", line.sku, f"{line.sku} is not in the product catalog")
             for line in inv.lines if line.sku not in known_skus]
    if po is None:
        found.append(Discrepancy("wrong_po_number", None,
                                 f"no purchase order matches {inv.po_number!r}"))
        return found

    for line in inv.lines:
        ordered = po.lines.get(line.sku)
        if line.sku not in known_skus:
            continue  # already reported as unknown_sku
        if ordered is None:
            found.append(Discrepancy("extra_line", line.sku,
                                     f"{line.sku} is not on {po.po_number}"))
            continue
        if abs(line.qty - ordered.qty) > TOL:
            found.append(Discrepancy("quantity_mismatch", line.sku,
                                     f"invoiced {line.qty}, ordered {ordered.qty}"))
        if abs(line.unit_price - ordered.unit_price) > TOL:
            found.append(Discrepancy("price_mismatch", line.sku,
                                     f"invoiced at {line.unit_price}, "
                                     f"ordered at {ordered.unit_price}"))

    invoiced = {line.sku for line in inv.lines}
    for sku in po.lines:
        if sku not in invoiced:
            found.append(Discrepancy("missing_line", sku,
                                     f"{sku} is on {po.po_number} but not invoiced"))
    return found


def check(inv: Invoice, po: PurchaseOrder | None, known_skus: set[str]) -> list[Discrepancy]:
    """Every deterministic check: reconciliation against the PO, then the invoice's own
    arithmetic. The result goes to the review queue, never back to the model."""
    found = reconcile(inv, po, known_skus)
    issues = arithmetic_issues(inv)
    if issues:
        found.append(Discrepancy("totals_mismatch", None, "; ".join(issues)))
    return found
