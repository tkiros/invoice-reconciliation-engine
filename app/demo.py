"""`make demo`: the whole flow without an API key, in ORACLE mode.

  1. process six chosen invoices
  2. upload one of them again (skipped: same sha256)
  3. show the review queue and why each document is there
  4. correct a misread as a reviewer, which re-runs the checks
  5. approve it and print its audit trail

No model is called. See app/oracle.py for exactly what the oracle returns.
"""

import hashlib
import os
from pathlib import Path

from app import oracle, pipeline, store
from app.schema import Invoice
from common import db

ROOT = Path(__file__).resolve().parent.parent
REVIEWER = "demo-reviewer"
BANNER = """\
==============================================================================
 OFFLINE DEMO, ORACLE MODE. No model is called. "Extraction" returns the
 generator's ground truth plus deliberate misreads (app/oracle.py), so that
 reconciliation, review routing and the audit trail can run without an API key.
 NOTHING BELOW MEASURES MODEL ACCURACY. For that, see `make eval` (needs a key).
=============================================================================="""


def kinds(row: dict) -> list[str]:
    return [d["kind"] for d in row["injected"]]


def pick_examples() -> list[tuple[str, dict]]:
    """One invoice per scenario, chosen by rule so the demo is the same on every machine."""
    rows, misread = oracle.truth_rows(), oracle.misread_for()

    def first(test) -> dict:
        return next(r for r in rows if test(r) and not r["prompt_injection"])

    return [
        ("clean digital invoice",
         first(lambda r: not r["injected"] and not r["scanned"])),
        ("vendor billed a higher price than the PO",
         first(lambda r: kinds(r) == ["price_mismatch"] and not r["scanned"])),
        ("printed totals don't add up",
         first(lambda r: kinds(r) == ["totals_mismatch"])),
        ("PO number has a typo",
         first(lambda r: kinds(r) == ["wrong_po_number"])),
        ("scan where the oracle misreads a SKU",
         first(lambda r: misread.get(r["id"]) == "sku" and not r["injected"])),
        ("scan where the oracle misreads the date",
         first(lambda r: misread.get(r["id"]) == "date" and not r["injected"])),
    ]


def show_document(conn, doc_id: int, label: str) -> None:
    doc = store.get_document(conn, doc_id)
    print(f"\n  [{doc_id}] {doc['filename']}: {label}")
    print(f"      status: {doc['status']}   review reasons: "
          f"{', '.join(doc['review_reasons']) or 'none'}")
    for d in store.discrepancies_for(conn, doc_id):
        print(f"      - {d['kind']}: {d['detail']}")


def main() -> None:
    os.environ["EXTRACTOR"] = "oracle"  # pipeline.choose_extractor() reads this
    print(BANNER)
    examples = pick_examples()
    files = {row["id"]: (ROOT / row["file"]).read_bytes() for _, row in examples}

    with db.connect() as conn:
        store.delete_documents(conn, [hashlib.sha256(b).hexdigest() for b in files.values()])
        print("\n1. Process six invoices")
        ids = {}
        for label, row in examples:
            ids[row["id"]], _ = pipeline.process_file(conn, files[row["id"]], f"{row['id']}.pdf")
            show_document(conn, ids[row["id"]], label)

        first_id = examples[0][1]["id"]
        print(f"\n2. Upload {first_id}.pdf again, under a new name")
        doc_id, is_new = pipeline.process_file(conn, files[first_id], "copy-of-invoice.pdf")
        print(f"   -> document {doc_id}, new: {is_new} (same sha256, so not processed or paid "
              "for twice)")

        print("\n3. Review queue")
        for doc in store.list_documents(conn, status="needs_review"):
            print(f"   [{doc['id']}] {doc['filename']}: {', '.join(doc['review_reasons'])}")

        label, row = examples[4]
        doc_id = ids[row["id"]]
        print(f"\n4. {REVIEWER} opens [{doc_id}], compares it with the PDF, fixes the SKU")
        corrected = Invoice.model_validate(row["expected"])  # what the PDF really says
        for field, old, new in pipeline.correct(conn, doc_id, corrected, REVIEWER):
            print(f"   corrected {field}: {old!r} -> {new!r}")
        show_document(conn, doc_id, "after the correction, checks re-run")

        print(f"\n5. {REVIEWER} approves [{doc_id}]. Its audit trail:")
        pipeline.approve(conn, doc_id, REVIEWER)
        for a in store.audit_trail(conn, doc_id):
            change = f"{a['field']}: {a['old_value']!r} -> {a['new_value']!r}" if a["field"] else ""
            print(f"   {a['created_at']:%H:%M:%S}  {a['actor']:<14} {a['action']:<16} {change}")

        date_row = examples[5][1]
        print(f"\nNote: [{ids[date_row['id']]}] has a wrong invoice date yet was auto-approved. A "
              "misread that breaks no rule\npasses every deterministic check; field accuracy in "
              "the eval measures how often that happens.")
    print("\nClick through it: EXTRACTOR=oracle make run, then open http://localhost:8002/review")


if __name__ == "__main__":
    main()
