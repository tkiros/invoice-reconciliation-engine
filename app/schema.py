"""The shape the model must return, and the arithmetic check that runs after it.

Both are copied from the build guide unchanged. The schema checks shape and types only;
arithmetic_issues() runs after extraction and its result is NEVER sent back to the model,
because a model told "the totals don't add up" will change numbers until they do.
"""
# ruff: noqa: E741  (the guide names a line `l`; kept so this matches the guide exactly)

from datetime import date
from decimal import Decimal

from pydantic import BaseModel

TOL = Decimal("0.01")


class Line(BaseModel):
    sku: str
    description: str
    qty: Decimal
    unit_price: Decimal
    line_total: Decimal


class Invoice(BaseModel):          # shape/type only -- what the model must return
    invoice_number: str
    po_number: str | None
    vendor: str
    invoice_date: date
    lines: list[Line]
    subtotal: Decimal
    tax: Decimal
    total: Decimal


def arithmetic_issues(inv: Invoice) -> list[str]:
    """Run AFTER extraction. Never fed back to the model."""
    issues = [f"{l.sku}: qty*unit_price != line_total"
              for l in inv.lines if abs(l.qty * l.unit_price - l.line_total) > TOL]
    if abs(sum(l.line_total for l in inv.lines) - inv.subtotal) > TOL:
        issues.append("lines don't sum to subtotal")
    if abs(inv.subtotal + inv.tax - inv.total) > TOL:
        issues.append("subtotal + tax != total")
    return issues
