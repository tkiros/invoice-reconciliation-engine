"""The scoring path, from hand-checked arithmetic up to full eval runs on a fake API.

The expected numbers below are written out by hand from the dev split's make-up:
50 invoices (38 clean, 12 scanned; 57 pages), 15 with an injected discrepancy
(12 clean, 3 scanned), 2 of them totals_mismatch (both clean).
"""

import hashlib
import json

import pytest

from app import oracle
from evals import run
from evals import score as scoring

TRUTH = {
    "invoice_number": "INV-1", "po_number": "PO-1", "vendor": "Acme Ltd",
    "invoice_date": "2026-03-14", "subtotal": "30.00", "tax": "3.00", "total": "33.00",
    "lines": [{"sku": "A-1", "description": "a", "qty": "2", "unit_price": "10.00",
               "line_total": "20.00"},
              {"sku": "B-2", "description": "b", "qty": "1", "unit_price": "10.00",
               "line_total": "10.00"}],
}


def test_pct_nearest_rank():
    assert run.pct([10, 20, 30, 40], 0.5) == 20
    assert run.pct([10, 20, 30, 40], 0.95) == 40
    assert run.pct([5], 0.95) == 5


def test_score_counts_a_partly_wrong_prediction():
    row = {"expected": TRUTH, "injected": [{"kind": "price_mismatch", "sku": "B-2"}]}
    pred = {
        "invoice": TRUTH | {
            "invoice_number": "INV-7",  # wrong
            "vendor": "ACME   ltd",  # right after normalising case and spaces
            "subtotal": 30,  # right: 30 == 30.00
            "total": "34.00",  # wrong
            "lines": [{"sku": "A-1", "qty": "2", "unit_price": "10"},  # right: 10 == 10.00
                      {"sku": "B-2", "qty": "1", "unit_price": "11.00"},  # wrong price
                      {"sku": "C-3", "qty": "1", "unit_price": "1.00"}],  # not on the invoice
        },
        "discrepancies": [["price_mismatch", "B-2"], ["extra_line", "C-3"]],
        "auto": False,
    }
    assert scoring.score(pred, row) == {
        "docs": 1, "fields_ok": 5, "fields": 7,
        "line_tp": 1, "line_pred": 3, "line_true": 2,
        "found_tp": 1, "found_pred": 2, "found_true": 1,
        "auto": 0, "bad": 1, "bad_auto": 0,
    }


def test_a_crash_scores_zero_everywhere():
    row = {"expected": TRUTH, "injected": []}
    assert scoring.score(None, row) == {
        "docs": 1, "fields_ok": 0, "fields": 7, "line_tp": 0, "line_pred": 0, "line_true": 2,
        "found_tp": 0, "found_pred": 0, "found_true": 0, "auto": 0, "bad": 0, "bad_auto": 0}


def test_aggregate_sums_counts_before_dividing():
    wrong = {"docs": 1, "fields_ok": 5, "fields": 7, "line_tp": 1, "line_pred": 3,
             "line_true": 2, "found_tp": 1, "found_pred": 2, "found_true": 1, "auto": 0,
             "bad": 1, "bad_auto": 0}
    crashed = {"docs": 1, "fields_ok": 0, "fields": 7, "line_tp": 0, "line_pred": 0,
               "line_true": 1, "found_tp": 0, "found_pred": 0, "found_true": 0, "auto": 0,
               "bad": 0, "bad_auto": 0}
    perfect = {"docs": 1, "fields_ok": 7, "fields": 7, "line_tp": 1, "line_pred": 1,
               "line_true": 1, "found_tp": 0, "found_pred": 0, "found_true": 0, "auto": 1,
               "bad": 0, "bad_auto": 0}
    assert scoring.aggregate([wrong, crashed, perfect]) == {
        "field_accuracy": 12 / 21,
        "line_f1": 4 / 8,  # TP 2, predicted 4, true 4
        "discrepancy_precision": 1 / 2,
        "discrepancy_recall": 1 / 1,
        "auto_processed": 1 / 3,
        "bad_auto_processed": 0 / 1,
    }
    nothing_flagged = scoring.aggregate([crashed])
    assert nothing_flagged["discrepancy_precision"] is None  # undefined, shown as n/a
    assert nothing_flagged["bad_auto_processed"] is None


# --- full eval runs against the fake API -------------------------------------------------

def truth_for(pdf: bytes) -> dict:
    return json.loads(json.dumps(
        oracle.truth_by_hash()[hashlib.sha256(pdf).hexdigest()]["expected"]))


def run_eval(tmp_path, *args: str) -> tuple[int, str]:
    out = tmp_path / "results.md"
    code = run.main(["--split", "dev", "--out", str(out), *args])
    return code, out.read_text()


def test_perfect_fake_scores_1_and_cost_comes_from_llm_calls(db, fake_claude, tmp_path):
    fake_claude(lambda pdf: json.dumps(truth_for(pdf)))
    code, report = run_eval(tmp_path)
    assert code == 0
    for line in [
        # System | Docs | Field acc | Line F1 | Disc P | Disc R | Auto | Bad auto | Errors
        "| pipeline | all (50) | 1.000 | 1.000 | 1.000 | 1.000 | 0.700 | 0.000 | 0 |",
        "| pipeline | clean (38) | 1.000 | 1.000 | 1.000 | 1.000 | 0.684 | 0.000 | 0 |",
        "| pipeline | scanned (12) | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.000 | 0 |",
        # The baseline has no arithmetic check, so it misses both totals_mismatch cases
        # (recall 13/15 and 10/12), and it auto-processes every bad invoice.
        "| baseline | all (50) | 1.000 | 1.000 | 1.000 | 0.867 | 1.000 | 1.000 | 0 |",
        "| baseline | clean (38) | 1.000 | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 | 0 |",
        "| baseline | scanned (12) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |",
        # One call per invoice at 1,000 in / 100 out on Haiku 5.5 = $0.00015; 57 pages.
        "| pipeline | all (50) | 50 | $0.000150 | $0.000132 | 0.0 s | 0.0 s |",
        "| pipeline | scanned (12) | 12 | $0.000150 | $0.000138 | 0.0 s | 0.0 s |",
    ]:
        assert line in report, line


def test_misread_fields_lower_field_accuracy_by_exactly_that_much(db, fake_claude, tmp_path):
    """Vendor wrong on every invoice, date also wrong on scans: 6/7 clean, 5/7 scanned,
    (38*6 + 12*5) / (50*7) = 0.823 overall. Neither field affects reconciliation."""
    def misread(pdf):
        data = truth_for(pdf)
        data["vendor"] = "Wrong Vendor Inc"
        if oracle.truth_by_hash()[hashlib.sha256(pdf).hexdigest()]["scanned"]:
            data["invoice_date"] = "2000-01-01"
        return json.dumps(data)

    fake_claude(misread)
    code, report = run_eval(tmp_path, "--systems", "pipeline")
    assert code == 0
    assert "| pipeline | all (50) | 0.823 | 1.000 | 1.000 | 1.000 | 0.700 | 0.000 | 0 |" in report
    assert "| pipeline | clean (38) | 0.857 |" in report
    assert "| pipeline | scanned (12) | 0.714 |" in report


def test_schema_failures_retries_and_baseline_crashes_are_counted(db, fake_claude, tmp_path):
    """Every reply has total '$...': the pipeline retries twice per invoice then routes it
    to review; the baseline tries once and crashes. First 5 dev invoices: no discrepancies."""
    def dollar_total(pdf):
        data = truth_for(pdf)
        return json.dumps(data | {"total": "$" + data["total"]})

    fake_claude(dollar_total)
    code, report = run_eval(tmp_path, "--limit", "5")
    assert code == 0  # baseline crashes are results, not harness failures
    assert "| pipeline | all (5) | 0.000 | 0.000 | n/a | n/a | 0.000 | n/a | 0 |" in report
    assert "| baseline | all (5) | 0.000 | 0.000 | n/a | n/a | 0.000 | n/a | 5 |" in report
    assert "| pipeline | all (5) | 15 | $0.000450 |" in report
    assert "| baseline | all (5) | 5 | $0.000150 |" in report


def test_eval_refuses_to_start_without_credentials(db, monkeypatch, tmp_path):
    monkeypatch.setattr(run, "has_credentials", lambda: False)
    with pytest.raises(SystemExit, match="No Anthropic credentials"):
        run_eval(tmp_path)
