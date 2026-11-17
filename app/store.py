"""Every SQL statement the app runs. Each function takes an open psycopg connection."""

from collections.abc import Iterable

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.reconcile import Discrepancy, POLine, PurchaseOrder
from app.schema import Invoice

SYSTEM = "system"  # the audit-log actor for automatic steps


# --- reference data (the ERP's side) ----------------------------------------------------

def load_erp(conn: psycopg.Connection, erp: dict) -> None:
    """Insert or update products and purchase orders. Safe to run again."""
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO products (sku, description) VALUES (%(sku)s, %(description)s)"
            " ON CONFLICT (sku) DO UPDATE SET description = EXCLUDED.description",
            erp["products"])
        for po in erp["purchase_orders"]:
            cur.execute(
                "INSERT INTO purchase_orders (po_number, vendor, po_date)"
                " VALUES (%(po_number)s, %(vendor)s, %(po_date)s) ON CONFLICT (po_number)"
                " DO UPDATE SET vendor = EXCLUDED.vendor, po_date = EXCLUDED.po_date", po)
            cur.execute("DELETE FROM po_lines WHERE po_number = %s", (po["po_number"],))
            cur.executemany(
                "INSERT INTO po_lines (po_number, sku, qty, unit_price)"
                " VALUES (%(po_number)s, %(sku)s, %(qty)s, %(unit_price)s)",
                [{**line, "po_number": po["po_number"]} for line in po["lines"]])


def find_po(conn: psycopg.Connection, po_number: str | None) -> PurchaseOrder | None:
    if not po_number:
        return None
    head = conn.execute("SELECT po_number, vendor FROM purchase_orders WHERE po_number = %s",
                        (po_number.strip(),)).fetchone()
    if head is None:
        return None
    rows = conn.execute("SELECT sku, qty, unit_price FROM po_lines WHERE po_number = %s",
                        (head[0],)).fetchall()
    lines = {sku: POLine(sku, qty, price) for sku, qty, price in rows}
    return PurchaseOrder(head[0], head[1], lines)


def known_skus(conn: psycopg.Connection, skus: Iterable[str]) -> set[str]:
    """The subset of `skus` that exists in the product catalog."""
    rows = conn.execute("SELECT sku FROM products WHERE sku = ANY(%s)", (list(skus),)).fetchall()
    return {sku for (sku,) in rows}


# --- documents ---------------------------------------------------------------------------

def insert_document(conn: psycopg.Connection, sha256: str, filename: str,
                    pdf: bytes) -> int | None:
    """A new row in status 'received', or None when these exact bytes were stored before.

    ON CONFLICT makes the duplicate check and the insert one atomic statement, so two
    simultaneous uploads of the same file can't both get through.
    """
    row = conn.execute(
        "INSERT INTO documents (sha256, filename, pdf, status) VALUES (%s, %s, %s, 'received')"
        " ON CONFLICT (sha256) DO NOTHING RETURNING id", (sha256, filename, pdf)).fetchone()
    if row is None:
        return None
    audit(conn, row[0], SYSTEM, "status_changed", "status", None, "received")
    return row[0]


def document_id_for(conn: psycopg.Connection, sha256: str) -> int | None:
    row = conn.execute("SELECT id FROM documents WHERE sha256 = %s", (sha256,)).fetchone()
    return row[0] if row else None


def save_extraction(conn: psycopg.Connection, doc_id: int, invoice: Invoice | None,
                    extractor: str, pages: int | None, error: str | None) -> None:
    conn.execute(
        "UPDATE documents SET invoice = %s, extractor = %s, pages = %s, error = %s,"
        " updated_at = now() WHERE id = %s",
        (as_json(invoice), extractor, pages, error, doc_id))


def save_invoice(conn: psycopg.Connection, doc_id: int, invoice: Invoice) -> None:
    conn.execute("UPDATE documents SET invoice = %s, updated_at = now() WHERE id = %s",
                 (as_json(invoice), doc_id))


def save_checks(conn: psycopg.Connection, doc_id: int, found: list[Discrepancy],
                reasons: list[str], actor: str = SYSTEM) -> None:
    """Replace the document's discrepancies and review reasons; audit a change of reasons."""
    old = conn.execute("SELECT review_reasons FROM documents WHERE id = %s",
                       (doc_id,)).fetchone()[0]
    conn.execute("DELETE FROM discrepancies WHERE document_id = %s", (doc_id,))
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO discrepancies (document_id, kind, sku, detail) VALUES (%s, %s, %s, %s)",
            [(doc_id, d.kind, d.sku, d.detail) for d in found])
    conn.execute("UPDATE documents SET review_reasons = %s, updated_at = now() WHERE id = %s",
                 (reasons, doc_id))
    if old != reasons:
        audit(conn, doc_id, actor, "reasons_changed", "review_reasons", ",".join(old),
              ",".join(reasons))


def set_status(conn: psycopg.Connection, doc_id: int, status: str, actor: str = SYSTEM) -> None:
    """The only way a status changes, so every change gets its audit row."""
    old = conn.execute("SELECT status FROM documents WHERE id = %s FOR UPDATE",
                       (doc_id,)).fetchone()[0]
    conn.execute("UPDATE documents SET status = %s, updated_at = now() WHERE id = %s",
                 (status, doc_id))
    audit(conn, doc_id, actor, "status_changed", "status", old, status)


def audit(conn: psycopg.Connection, doc_id: int, actor: str, action: str,
          field: str | None = None, old: str | None = None, new: str | None = None) -> None:
    conn.execute(
        "INSERT INTO audit_log (document_id, actor, action, field, old_value, new_value)"
        " VALUES (%s, %s, %s, %s, %s, %s)", (doc_id, actor, action, field, old, new))


def delete_documents(conn: psycopg.Connection, sha256s: list[str]) -> None:
    """Dev resets only (`make demo` clears its own earlier run). The app never deletes."""
    conn.execute("DELETE FROM documents WHERE sha256 = ANY(%s)", (sha256s,))


def as_json(invoice: Invoice | None) -> Jsonb | None:
    """Decimals and dates as strings, so nothing is lost in JSON."""
    return Jsonb(invoice.model_dump(mode="json")) if invoice else None


# --- reads for the review UI and the demo -----------------------------------------------

def get_document(conn: psycopg.Connection, doc_id: int) -> dict | None:
    """One document without its PDF bytes."""
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT id, sha256, filename, pages, status, extractor, invoice, review_reasons,"
            " error, created_at FROM documents WHERE id = %s", (doc_id,)).fetchone()


def get_pdf(conn: psycopg.Connection, doc_id: int) -> bytes | None:
    row = conn.execute("SELECT pdf FROM documents WHERE id = %s", (doc_id,)).fetchone()
    return bytes(row[0]) if row else None


def list_documents(conn: psycopg.Connection, status: str | None = None) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT id, filename, status, extractor, review_reasons, created_at FROM documents"
            " WHERE %(status)s::text IS NULL OR status = %(status)s ORDER BY id",
            {"status": status}).fetchall()


def discrepancies_for(conn: psycopg.Connection, doc_id: int) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute("SELECT kind, sku, detail FROM discrepancies WHERE document_id = %s"
                           " ORDER BY id", (doc_id,)).fetchall()


def audit_trail(conn: psycopg.Connection, doc_id: int) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT created_at, actor, action, field, old_value, new_value FROM audit_log"
            " WHERE document_id = %s ORDER BY id", (doc_id,)).fetchall()


def llm_cost(conn: psycopg.Connection, sha256: str) -> tuple[int, float | None]:
    """(model calls, total measured cost) for one document; cost None if any call is unpriced."""
    calls, total, unpriced = conn.execute(
        "SELECT count(*), coalesce(sum(cost_usd), 0), count(*) FILTER (WHERE cost_usd IS NULL)"
        " FROM llm_calls WHERE ref = %s", (sha256,)).fetchone()
    return calls, (None if unpriced else float(total))
