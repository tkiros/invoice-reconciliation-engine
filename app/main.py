"""HTTP API and the review page.

  POST /documents          upload a PDF (JSON reply)
  GET  /review             the review queue, with an upload form
  GET  /review/{id}        the PDF next to its extracted fields; correct or approve
  GET  /documents/{id}/pdf the stored PDF

Plain HTML from f-strings, every value passed through html.escape. No login: the reviewer
types their name, which goes into the audit log (see Limitations in the README).
"""

import os
from html import escape
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import ValidationError

from app import oracle, pipeline, store
from app.schema import Invoice
from common import db

app = FastAPI(title="Invoice reconciliation")

HEADER_FIELDS = ["invoice_number", "po_number", "vendor", "invoice_date", "subtotal", "tax",
                 "total"]
LINE_FIELDS = ["sku", "description", "qty", "unit_price", "line_total"]
CSS = """body{font-family:system-ui,sans-serif;margin:1.5rem;color:#222}
table{border-collapse:collapse;margin:.5rem 0}td,th{border:1px solid #ccc;padding:.25rem .5rem;
text-align:left}input{font:inherit;width:100%;box-sizing:border-box}.oracle{background:#b00020;
color:#fff;padding:.6rem;font-weight:bold}.cols{display:flex;gap:1.5rem}.cols>*{flex:1}
iframe{width:100%;height:85vh;border:1px solid #ccc}.bad{color:#b00020}"""


def connection():
    with db.connect() as conn:
        yield conn


Conn = Annotated[psycopg.Connection, Depends(connection)]


@app.get("/health")
def health(conn: Conn) -> dict[str, str]:
    conn.execute("SELECT 1")
    return {"status": "ok"}


def read_upload(file: UploadFile) -> bytes:
    pdf = file.file.read(pipeline.MAX_PDF_BYTES + 1)  # never read an unbounded body
    if len(pdf) > pipeline.MAX_PDF_BYTES:
        raise HTTPException(413, f"PDF is over {pipeline.MAX_PDF_BYTES // 2**20} MB")
    return pdf


@app.post("/documents")
def upload(conn: Conn, file: UploadFile) -> dict:
    """Process one PDF. Uploading the same bytes again returns the first result."""
    doc_id, is_new = pipeline.process_file(conn, read_upload(file), file.filename or "upload.pdf")
    doc = store.get_document(conn, doc_id)
    return {"id": doc_id, "duplicate": not is_new, "status": doc["status"],
            "review_reasons": doc["review_reasons"], "extractor": doc["extractor"]}


@app.get("/documents/{doc_id}/pdf")
def pdf(conn: Conn, doc_id: int) -> Response:
    data = store.get_pdf(conn, doc_id)
    if data is None:
        raise HTTPException(404)
    return Response(data, media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="document-{doc_id}.pdf"'})


# --- review UI -----------------------------------------------------------------------------

def page(title: str, body: str, status_code: int = 200) -> HTMLResponse:
    banner = ""
    if os.environ.get("EXTRACTOR") == "oracle":
        banner = (f'<p class="oracle">OFFLINE DEMO MODE: new uploads are "extracted" by the '
                  f"{escape(oracle.LABEL)}. Nothing here measures model accuracy.</p>")
    return HTMLResponse(f"<!doctype html><html><head><meta charset='utf-8'><title>"
                        f"{escape(title)}</title><style>{CSS}</style></head><body>{banner}"
                        f"<p><a href='/review'>Review queue</a></p><h1>{escape(title)}</h1>"
                        f"{body}</body></html>", status_code=status_code)


def reasons_html(reasons: list[str]) -> str:
    if not reasons:
        return "none"
    return "<ul>" + "".join(f"<li><b>{escape(r)}</b>: {escape(pipeline.REASONS.get(r, ''))}</li>"
                            for r in reasons) + "</ul>"


def table(headers: list[str], rows: list[list[str]]) -> str:
    """Cells are inserted as given: callers escape them."""
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def documents_table(docs: list[dict]) -> str:
    rows = [[f"<a href='/review/{d['id']}'>{d['id']}</a>", escape(d["filename"]),
             escape(d["status"]), escape(", ".join(d["review_reasons"]) or "-"),
             escape(d["extractor"] or "-")] for d in docs]
    return table(["Id", "File", "Status", "Review reasons", "Extracted by"], rows)


@app.get("/review", response_class=HTMLResponse)
def review_queue(conn: Conn) -> HTMLResponse:
    docs = store.list_documents(conn)
    waiting = [d for d in docs if d["status"] == "needs_review"]
    upload_form = ("<form method='post' action='/review/upload' enctype='multipart/form-data'>"
                   "<input type='file' name='file' accept='application/pdf' required "
                   "style='width:auto'> <button>Upload and process</button></form>")
    body = (f"{upload_form}<h2>Waiting for review: {len(waiting)}</h2>{documents_table(waiting)}"
            f"<h2>All documents: {len(docs)}</h2>{documents_table(docs)}")
    return page("Review queue", body)


@app.post("/review/upload")
def review_upload(conn: Conn, file: UploadFile) -> RedirectResponse:
    doc_id, _ = pipeline.process_file(conn, read_upload(file), file.filename or "upload.pdf")
    return RedirectResponse(f"/review/{doc_id}", status_code=303)


def field_input(name: str, value: object, editable: bool) -> str:
    shown = "" if value is None else str(value)
    disabled = "" if editable else " disabled"
    return f"<input name='{name}' value='{escape(shown, quote=True)}'{disabled}>"


def invoice_form(doc: dict, editable: bool) -> str:
    """Header fields, then one row per line, plus a blank row for adding a missed line."""
    invoice = doc["invoice"] or {"lines": []}
    header = table(["Field", "Value"], [[escape(f), field_input(f, invoice.get(f), editable)]
                                        for f in HEADER_FIELDS])
    lines = invoice["lines"] + ([{}] if editable else [])
    rows = []
    for i, line in enumerate(lines):
        cells = [field_input(f"{f}-{i}", line.get(f), editable) for f in LINE_FIELDS]
        if editable:
            cells.append(f"<input type='checkbox' name='remove-{i}' style='width:auto'>"
                         if line else "(new line)")
        rows.append(cells)
    headers = LINE_FIELDS + (["remove"] if editable else [])
    fields = header + "<h3>Lines</h3>" + table(headers, rows)
    if not editable:
        return fields
    name = ("<label>Your name (goes into the audit log) "
            "<input name='reviewer' required style='width:16rem'></label> ")
    # Two forms, so Approve can never silently drop edits that weren't saved.
    return (f"<form method='post' action='/review/{doc['id']}/correct'>{fields}"
            f"<p>{name}<button>Save corrections and re-check</button></p></form>"
            f"<form method='post' action='/review/{doc['id']}/approve'>"
            f"<p>{name}<button>Approve as saved</button></p></form>")


@app.get("/review/{doc_id}", response_class=HTMLResponse)
def review_document(conn: Conn, doc_id: int) -> HTMLResponse:
    doc = store.get_document(conn, doc_id)
    if doc is None:
        raise HTTPException(404)
    calls, cost = store.llm_cost(conn, doc["sha256"])
    found = store.discrepancies_for(conn, doc_id)
    trail = store.audit_trail(conn, doc_id)
    editable = doc["status"] == "needs_review"
    summary = (f"<p>Status: <b>{escape(doc['status'])}</b> · pages: {doc['pages']} · "
               f"extracted by: <b>{escape(doc['extractor'] or '-')}</b> · model calls: {calls}"
               f" · measured cost: {'unpriced' if cost is None else f'${cost:.6f}'}</p>"
               f"<p>Review reasons: {reasons_html(doc['review_reasons'])}</p>")
    if doc["error"]:
        summary += f"<p class='bad'>Error: {escape(doc['error'])}</p>"
    issues = table(["Kind", "SKU", "Detail"],
                   [[escape(d["kind"]), escape(d["sku"] or "-"), escape(d["detail"])]
                    for d in found]) if found else "<p>none</p>"
    history = table(["When (UTC)", "Who", "Action", "Field", "Old", "New"],
                    [[f"{a['created_at']:%Y-%m-%d %H:%M:%S}", escape(a["actor"]),
                      escape(a["action"]), escape(a["field"] or ""), escape(a["old_value"] or ""),
                      escape(a["new_value"] or "")] for a in trail])
    body = (summary + "<div class='cols'>"
            f"<div><iframe src='/documents/{doc_id}/pdf' title='invoice PDF'></iframe></div>"
            f"<div><h2>Extracted fields</h2>{invoice_form(doc, editable)}"
            f"<h2>Discrepancies</h2>{issues}</div></div><h2>Audit trail</h2>{history}")
    return page(f"Document {doc_id}: {doc['filename']}", body)


def invoice_from_form(form) -> Invoice:
    """Rebuild the invoice from the form. Blank and ticked-for-removal lines are dropped."""
    data = {f: str(form.get(f, "")).strip() for f in HEADER_FIELDS}
    data["po_number"] = data["po_number"] or None
    data["lines"] = []
    i = 0
    while f"sku-{i}" in form:
        line = {f: str(form.get(f"{f}-{i}", "")).strip() for f in LINE_FIELDS}
        if any(line.values()) and not form.get(f"remove-{i}"):
            data["lines"].append(line)
        i += 1
    return Invoice.model_validate(data)


def reviewer_from(form) -> str:
    reviewer = str(form.get("reviewer", "")).strip()
    if not reviewer:
        raise HTTPException(422, "Enter your name: it goes into the audit log.")
    return reviewer


# async only to read the form; the work after it is quick database calls, not model calls.
@app.post("/review/{doc_id}/correct")
async def correct(conn: Conn, doc_id: int, request: Request) -> Response:
    form = await request.form()
    reviewer = reviewer_from(form)
    try:
        corrected = invoice_from_form(form)
    except ValidationError as e:
        return page("Correction not saved", f"<pre>{escape(str(e))}</pre>"
                    f"<p><a href='/review/{doc_id}'>Back to the document</a></p>",
                    status_code=422)
    try:
        pipeline.correct(conn, doc_id, corrected, reviewer)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    return RedirectResponse(f"/review/{doc_id}", status_code=303)


@app.post("/review/{doc_id}/approve")
async def approve(conn: Conn, doc_id: int, request: Request) -> RedirectResponse:
    form = await request.form()
    try:
        pipeline.approve(conn, doc_id, reviewer_from(form))
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    return RedirectResponse(f"/review/{doc_id}", status_code=303)
