"""Command line.

    uv run python -m app.cli seed [data/generated/erp.json]   load products and POs (make seed)
    uv run python -m app.cli process PATH                     one PDF, or every PDF in a folder

Set EXTRACTOR=oracle to process without an API key (offline demo mode, not a model).
"""

import argparse
import json
from pathlib import Path

from app import pipeline, store
from common import db

ERP_FILE = Path("data/generated/erp.json")


def seed(path: Path) -> None:
    erp = json.loads(path.read_text(encoding="utf-8"))
    with db.connect() as conn:
        store.load_erp(conn, erp)
    print(f"loaded {len(erp['products'])} products and "
          f"{len(erp['purchase_orders'])} purchase orders from {path}")


def pdf_paths(target: Path) -> list[Path]:
    return sorted(target.glob("*.pdf")) if target.is_dir() else [target]


def process(target: Path) -> None:
    label, _ = pipeline.choose_extractor()
    print(f"extractor: {label}")
    with db.connect() as conn:
        for path in pdf_paths(target):
            doc_id, is_new = pipeline.process_file(conn, path.read_bytes(), path.name)
            doc = store.get_document(conn, doc_id)
            reasons = ", ".join(doc["review_reasons"]) or "-"
            skipped = "" if is_new else "  (same file as before: not processed again)"
            print(f"{path.name}: document {doc_id}, {doc['status']}, reasons: {reasons}{skipped}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Invoice reconciliation")
    commands = parser.add_subparsers(dest="command", required=True)
    seed_cmd = commands.add_parser("seed", help="load products and purchase orders")
    seed_cmd.add_argument("path", nargs="?", type=Path, default=ERP_FILE)
    process_cmd = commands.add_parser("process", help="process a PDF or a folder of PDFs")
    process_cmd.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "seed":
        seed(args.path)
    else:
        process(args.path)


if __name__ == "__main__":
    main()
