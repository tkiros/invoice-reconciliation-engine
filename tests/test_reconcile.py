"""One test per discrepancy type, plus a check over the whole generated dataset."""

from decimal import Decimal

from app.reconcile import POLine, PurchaseOrder, check, reconcile
from app.schema import Invoice
from data import generate

CATALOG = {"AB-1000", "AB-2000", "AB-3000"}
PO = PurchaseOrder("PO-100001", "Acme", {
    "AB-1000": POLine("AB-1000", Decimal("2"), Decimal("10.00")),
    "AB-2000": POLine("AB-2000", Decimal("1"), Decimal("5.50")),
})


def invoice(lines: list[tuple[str, str, str]], po_number: str = "PO-100001",
            total_offset: str = "0") -> Invoice:
    """Build a self-consistent invoice from (sku, qty, unit_price) rows, tax 0."""
    rows = [{"sku": sku, "description": "x", "qty": qty, "unit_price": price,
             "line_total": str(Decimal(qty) * Decimal(price))} for sku, qty, price in lines]
    subtotal = sum(Decimal(r["line_total"]) for r in rows)
    return Invoice.model_validate({
        "invoice_number": "INV-1", "po_number": po_number, "vendor": "Acme",
        "invoice_date": "2026-03-14", "lines": rows, "subtotal": str(subtotal), "tax": "0",
        "total": str(subtotal + Decimal(total_offset))})


def kinds(found) -> list[tuple[str, str | None]]:
    return [(d.kind, d.sku) for d in found]


def test_matching_invoice_has_no_discrepancies():
    inv = invoice([("AB-1000", "2", "10.00"), ("AB-2000", "1", "5.50")])
    assert check(inv, PO, CATALOG) == []


def test_price_mismatch():
    inv = invoice([("AB-1000", "2", "11.00"), ("AB-2000", "1", "5.50")])
    found = reconcile(inv, PO, CATALOG)
    assert kinds(found) == [("price_mismatch", "AB-1000")]
    assert found[0].detail == "invoiced at 11.00, ordered at 10.00"


def test_price_within_one_cent_is_not_a_mismatch():
    inv = invoice([("AB-1000", "2", "10.01"), ("AB-2000", "1", "5.50")])
    assert reconcile(inv, PO, CATALOG) == []


def test_quantity_mismatch():
    inv = invoice([("AB-1000", "3", "10.00"), ("AB-2000", "1", "5.50")])
    assert kinds(reconcile(inv, PO, CATALOG)) == [("quantity_mismatch", "AB-1000")]


def test_missing_line():
    inv = invoice([("AB-1000", "2", "10.00")])
    assert kinds(reconcile(inv, PO, CATALOG)) == [("missing_line", "AB-2000")]


def test_extra_line():
    inv = invoice([("AB-1000", "2", "10.00"), ("AB-2000", "1", "5.50"), ("AB-3000", "1", "1")])
    assert kinds(reconcile(inv, PO, CATALOG)) == [("extra_line", "AB-3000")]


def test_wrong_po_number():
    inv = invoice([("AB-1000", "2", "10.00")], po_number="PO-100009")
    assert kinds(reconcile(inv, None, CATALOG)) == [("wrong_po_number", None)]


def test_totals_that_dont_add_up():
    inv = invoice([("AB-1000", "2", "10.00"), ("AB-2000", "1", "5.50")], total_offset="10")
    assert reconcile(inv, PO, CATALOG) == []  # reconciliation alone can't see it
    found = check(inv, PO, CATALOG)
    assert kinds(found) == [("totals_mismatch", None)]
    assert found[0].detail == "subtotal + tax != total"


def test_unknown_sku_is_reported_once_and_the_ordered_line_is_missing():
    """A misread SKU (AB1000 for AB-1000) is not also called an extra line."""
    inv = invoice([("AB1000", "2", "10.00"), ("AB-2000", "1", "5.50")])
    assert kinds(reconcile(inv, PO, CATALOG)) == [("unknown_sku", "AB1000"),
                                                  ("missing_line", "AB-1000")]


def test_unknown_sku_is_still_reported_when_the_po_is_not_found():
    inv = invoice([("ZZ-0000", "1", "1.00")], po_number="PO-999999")
    assert kinds(reconcile(inv, None, CATALOG)) == [("unknown_sku", "ZZ-0000"),
                                                    ("wrong_po_number", None)]


def test_perfect_extraction_finds_exactly_the_injected_discrepancies():
    """Over all 200 generated pairs: with the true invoice as input, the checks report
    every injected discrepancy and nothing else. So any eval error is the extraction's."""
    catalog, pairs = generate.build_dataset()
    known = {p["sku"] for p in catalog}
    pos = {p["po"]["po_number"]: PurchaseOrder(
        p["po"]["po_number"], p["po"]["vendor"],
        {line["sku"]: POLine(line["sku"], line["qty"], line["unit_price"])
         for line in p["po"]["lines"]}) for p in pairs}
    for pair in pairs:
        inv = Invoice.model_validate(pair["invoice"])
        found = check(inv, pos.get(inv.po_number), known)
        expected = [(d["kind"], d["sku"]) for d in pair["injected"]]
        assert kinds(found) == expected, pair["id"]
