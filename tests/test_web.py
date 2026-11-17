"""The HTTP API and review page, through FastAPI's test client, in oracle mode (no model)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import oracle, pipeline, store
from app.main import app

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client(db, monkeypatch):
    monkeypatch.setenv("EXTRACTOR", "oracle")
    return TestClient(app)


def row_where(test) -> dict:
    return next(r for r in oracle.truth_rows() if test(r))


def misread_sku_row() -> dict:
    """A scan the oracle misreads (SKU loses its dash), so it lands in review."""
    misread = oracle.misread_for()
    return row_where(lambda r: misread.get(r["id"]) == "sku" and not r["injected"])


def upload(client, row: dict, name: str | None = None):
    pdf = (ROOT / row["file"]).read_bytes()
    return client.post("/documents", files={"file": (name or f"{row['id']}.pdf", pdf,
                                                     "application/pdf")})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_twice_returns_the_first_result(client):
    row = row_where(lambda r: not r["injected"] and not r["scanned"])
    first = upload(client, row).json()
    assert first["status"] == "auto_approved" and first["duplicate"] is False
    assert first["extractor"] == oracle.LABEL
    again = upload(client, row, "other-name.pdf").json()
    assert (again["id"], again["duplicate"]) == (first["id"], True)


def test_review_page_shows_reasons_pdf_and_oracle_banner(client):
    row = misread_sku_row()
    doc = upload(client, row).json()
    assert doc["review_reasons"] == ["unknown_sku", "discrepancy"]

    queue = client.get("/review").text
    assert f"/review/{doc['id']}" in queue and "unknown_sku" in queue
    assert "OFFLINE DEMO MODE" in queue

    page = client.get(f"/review/{doc['id']}").text
    assert "a SKU is not in the product catalog" in page
    assert f"/documents/{doc['id']}/pdf" in page

    served = client.get(f"/documents/{doc['id']}/pdf")
    assert served.headers["content-type"] == "application/pdf"
    assert served.content == (ROOT / row["file"]).read_bytes()


def form_from(row: dict, reviewer: str) -> dict:
    """The review form, filled in with what the PDF really says."""
    expected = row["expected"]
    form = {f: expected[f] or "" for f in ("invoice_number", "po_number", "vendor",
                                            "invoice_date", "subtotal", "tax", "total")}
    for i, line in enumerate(expected["lines"]):
        for f in ("sku", "description", "qty", "unit_price", "line_total"):
            form[f"{f}-{i}"] = line[f]
    n = len(expected["lines"])  # the blank "add a line" row, left empty
    form.update({f"{f}-{n}": "" for f in ("sku", "description", "qty", "unit_price",
                                          "line_total")})
    return form | {"reviewer": reviewer}


def test_correct_then_approve_through_the_form(client, db):
    row = misread_sku_row()
    doc_id = upload(client, row).json()["id"]

    saved = client.post(f"/review/{doc_id}/correct", data=form_from(row, "carol"),
                        follow_redirects=False)
    assert saved.status_code == 303
    corrections = [a for a in store.audit_trail(db, doc_id) if a["action"] == "field_corrected"]
    sku = row["expected"]["lines"][0]["sku"]
    assert [(a["actor"], a["field"], a["old_value"], a["new_value"]) for a in corrections] == [
        ("carol", "lines[0].sku", sku.replace("-", ""), sku)]
    assert store.get_document(db, doc_id)["review_reasons"] == []

    approved = client.post(f"/review/{doc_id}/approve", data={"reviewer": "carol"},
                           follow_redirects=False)
    assert approved.status_code == 303
    assert store.get_document(db, doc_id)["status"] == "approved"
    assert client.post(f"/review/{doc_id}/approve", data={"reviewer": "carol"}).status_code == 409


def test_bad_form_input_is_rejected_without_saving(client, db):
    row = misread_sku_row()
    doc_id = upload(client, row).json()["id"]
    no_name = client.post(f"/review/{doc_id}/correct", data=form_from(row, ""))
    assert no_name.status_code == 422
    bad_amount = client.post(f"/review/{doc_id}/correct",
                             data=form_from(row, "carol") | {"total": "twelve"})
    assert bad_amount.status_code == 422
    assert "Correction not saved" in bad_amount.text
    assert not [a for a in store.audit_trail(db, doc_id) if a["actor"] == "carol"]


def test_lines_can_be_removed_and_added(client, db):
    row = misread_sku_row()
    doc_id = upload(client, row).json()["id"]
    form = form_from(row, "dan")
    n = len(row["expected"]["lines"])
    form |= {"remove-0": "on", f"sku-{n}": "XX-0001", f"description-{n}": "new",
             f"qty-{n}": "1", f"unit_price-{n}": "2.00", f"line_total-{n}": "2.00"}
    assert client.post(f"/review/{doc_id}/correct", data=form,
                       follow_redirects=False).status_code == 303
    skus = [line["sku"] for line in store.get_document(db, doc_id)["invoice"]["lines"]]
    assert skus == [line["sku"] for line in row["expected"]["lines"][1:]] + ["XX-0001"]


def test_filenames_are_escaped_in_html(client):
    row = row_where(lambda r: not r["injected"] and not r["scanned"])
    doc_id = upload(client, row, "<script>alert(1)</script>.pdf").json()["id"]
    page = client.get(f"/review/{doc_id}").text
    assert "<script>alert(1)" not in page
    assert "&lt;script&gt;" in page


def test_oversized_upload_is_refused(client, monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_PDF_BYTES", 10)
    response = client.post("/documents", files={"file": ("big.pdf", b"x" * 11)})
    assert response.status_code == 413
