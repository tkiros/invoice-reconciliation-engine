"""The dataset is what the README says it is, and the same on every run."""

import json
import random
from collections import Counter

import pypdfium2 as pdfium

from app.schema import Invoice, arithmetic_issues
from data import generate, render


def test_same_seed_same_dataset():
    assert generate.build_dataset() == generate.build_dataset()


def test_dataset_shape():
    _, pairs = generate.build_dataset()
    dev = [p for p in pairs if p["split"] == "dev"]
    assert (len(pairs), len(dev)) == (200, 50)
    discrepant = [p for p in pairs if p["injected"]]
    assert len(discrepant) == 60  # 30%
    assert sum(1 for p in dev if p["injected"]) == 15
    assert all(len(p["injected"]) == 1 for p in discrepant)
    kinds = Counter(p["injected"][0]["kind"] for p in discrepant)
    assert kinds == {kind: 10 for kind in generate.KINDS}
    assert sum(p["scanned"] for p in pairs) == 50  # 25%
    assert sum(p["prompt_injection"] for p in pairs) == generate.N_PROMPT_INJECTIONS
    assert {p["vendor"]["layout"] for p in pairs} == {"classic", "band", "european", "ledger"}
    assert max(len(p["invoice"]["lines"]) for p in pairs) >= 25  # some long invoices


def test_only_totals_mismatch_breaks_the_arithmetic():
    _, pairs = generate.build_dataset()
    for pair in pairs:
        issues = arithmetic_issues(Invoice.model_validate(pair["invoice"]))
        broken = [d for d in pair["injected"] if d["kind"] == "totals_mismatch"]
        assert bool(issues) == bool(broken), pair["id"]


def test_wrong_po_numbers_match_no_real_po():
    _, pairs = generate.build_dataset()
    real = {p["po"]["po_number"] for p in pairs}
    for pair in pairs:
        if pair["injected"] and pair["injected"][0]["kind"] == "wrong_po_number":
            assert pair["invoice"]["po_number"] not in real


def test_committed_ground_truth_matches_the_generator(generated):
    """evals/sets/*.jsonl is exactly what data/generate.py writes (no hand edits, no drift)."""
    catalog, pairs = generate.build_dataset()
    for split in ("dev", "test"):
        committed = (generate.SETS / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in committed]
        fresh = [p for p in pairs if p["split"] == split]
        assert len(rows) == len(fresh)
        for row, pair in zip(rows, fresh, strict=True):
            expected = json.loads(json.dumps(generate.truth_row(pair, row["pages"]),
                                             default=generate.to_json))
            assert row == expected


def test_rendering_is_reproducible_and_scans_have_no_text_layer():
    _, pairs = generate.build_dataset()
    pair = pairs[0]
    pdf, pages = render.invoice_pdf(pair["invoice"], pair["vendor"], generate.NOTE)
    assert render.invoice_pdf(pair["invoice"], pair["vendor"], generate.NOTE)[0] == pdf
    text = pdfium.PdfDocument(pdf)[0].get_textpage().get_text_range()
    assert pair["invoice"]["invoice_number"] in text

    scan = render.scan(pdf, random.Random("x"))
    assert render.scan(pdf, random.Random("x")) == scan
    scanned = pdfium.PdfDocument(scan)
    assert len(scanned) == pages
    assert scanned[0].get_textpage().get_text_range().strip() == ""


def test_money_and_date_styles():
    from datetime import date
    from decimal import Decimal
    value = Decimal("1234.5")
    assert [render.fmt_money(value, s) for s in ("us", "us_symbol", "eu", "plain")] == [
        "1,234.50", "$1,234.50", "1.234,50", "1234.50"]
    d = date(2026, 3, 4)
    assert [render.fmt_date(d, s) for s in ("iso", "us", "eu", "long")] == [
        "2026-03-04", "03/04/2026", "04.03.2026", "4 March 2026"]
