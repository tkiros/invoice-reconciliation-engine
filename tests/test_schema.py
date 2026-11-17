from decimal import Decimal

from app.schema import Invoice, arithmetic_issues


def invoice(**changes) -> Invoice:
    """2 x 10.00 + 1 x 5.50 = 25.50, tax 2.55, total 28.05; override any field."""
    data = {
        "invoice_number": "INV-1", "po_number": "PO-100001", "vendor": "Acme",
        "invoice_date": "2026-03-14",
        "lines": [
            {"sku": "AB-1000", "description": "bolt", "qty": "2", "unit_price": "10.00",
             "line_total": "20.00"},
            {"sku": "AB-2000", "description": "nut", "qty": "1", "unit_price": "5.50",
             "line_total": "5.50"},
        ],
        "subtotal": "25.50", "tax": "2.55", "total": "28.05",
    }
    return Invoice.model_validate(data | changes)


def test_consistent_invoice_has_no_issues():
    assert arithmetic_issues(invoice()) == []


def test_line_total_that_is_not_qty_times_price():
    inv = invoice()
    inv.lines[0].line_total = Decimal("21.00")
    issues = arithmetic_issues(inv)
    assert "AB-1000: qty*unit_price != line_total" in issues
    assert "lines don't sum to subtotal" in issues  # 21.00 + 5.50 != 25.50


def test_subtotal_that_is_not_the_sum_of_lines():
    assert arithmetic_issues(invoice(subtotal="26.50", total="29.05")) == [
        "lines don't sum to subtotal"]


def test_total_that_is_not_subtotal_plus_tax():
    assert arithmetic_issues(invoice(total="38.05")) == ["subtotal + tax != total"]


def test_one_cent_is_within_tolerance():
    assert arithmetic_issues(invoice(total="28.06")) == []
    assert arithmetic_issues(invoice(total="28.07")) == ["subtotal + tax != total"]


def test_model_output_with_amounts_as_numbers_or_strings_validates():
    """The structured-output schema allows a number or a string for each Decimal."""
    inv = Invoice.model_validate_json(
        '{"invoice_number": "1", "po_number": null, "vendor": "V", "invoice_date": "2026-01-02",'
        ' "lines": [], "subtotal": 0, "tax": "0.00", "total": 0.0}')
    assert inv.total == Decimal("0")
    assert inv.po_number is None
