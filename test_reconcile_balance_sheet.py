import contextlib
import io
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from reconcile_balance_sheet import (
    BalanceSheetDataError,
    main,
    print_report,
    reconcile_document,
)


def value(period_id, amount, status="reported"):
    return {
        "periodId": period_id,
        "status": status,
        "amount": amount,
        "sourceText": str(amount),
    }


def node(node_id, amount, operator="TOTAL", children=None, status="reported"):
    kind = (
        "total"
        if operator == "TOTAL"
        and (node_id == "Assets" or node_id.casefold().startswith("total "))
        else "subtotal"
        if operator == "TOTAL"
        else "line_item"
    )
    return {
        "nodeId": node_id,
        "kind": kind,
        "labels": {"ja": node_id, "en": node_id},
        "operator": operator,
        "values": [value("current", amount, status)],
        "children": children or [],
    }


def document(*items):
    return {
        "statement": {
            "periods": [
                {
                    "id": "current",
                    "kind": "current",
                    "date": "2026-03-31",
                    "labels": {"ja": "当期", "en": "Current period"},
                }
            ],
            "items": list(items),
        }
    }


class ReconcileBalanceSheetTests(unittest.TestCase):
    def test_corrected_sample_reconciles_using_display_unit_rounding(self):
        sample_path = Path(__file__).with_name("balance_sheet_after_correction.json")
        sample = json.loads(sample_path.read_text(encoding="utf-8"))

        report = reconcile_document(sample)

        self.assertEqual(len(report.reconciliations), 20)
        self.assertEqual(len(report.skipped), 0)
        self.assertEqual(
            sum(
                check.difference != 0
                and abs(check.difference) <= check.allowed_difference
                for check in report.reconciliations
            ),
            13,
        )
        self.assertEqual(report.inconsistencies, [])

    def test_empty_reconciliation_report_does_not_claim_success(self):
        output = io.StringIO()
        print_report(reconcile_document(document()), None, output)
        self.assertIn("No reconciliation checks were performed", output.getvalue())
        self.assertIn("was not verified", output.getvalue())

    def test_sums_add_and_subtract_children(self):
        parent = node(
            "Net property",
            197_775,
            children=[
                node("Gross property", 2_121_360, "ADD"),
                node("Accumulated depreciation", 1_923_585, "SUBTRACT"),
            ],
        )
        report = reconcile_document(document(parent))
        self.assertEqual(len(report.reconciliations), 1)
        self.assertEqual(report.reconciliations[0].expected, Decimal(197_775))
        self.assertEqual(report.inconsistencies, [])

    def test_subtract_operator_handles_already_negative_amount(self):
        parent = node(
            "Net property",
            197_775,
            children=[
                node("Gross property", 2_121_360, "ADD"),
                node("Accumulated depreciation", -1_923_585, "SUBTRACT"),
            ],
        )
        report = reconcile_document(document(parent))
        self.assertEqual(report.reconciliations[0].expected, Decimal(197_775))

    def test_reports_mismatch_and_difference(self):
        parent = node(
            "Assets",
            100,
            children=[
                node("Cash", 65, "ADD"),
                node("Receivables", 30, "ADD"),
            ],
        )
        report = reconcile_document(document(parent))
        self.assertEqual(len(report.inconsistencies), 1)
        self.assertEqual(report.inconsistencies[0].difference, Decimal(5))

    def test_ignores_total_rows_when_detail_rows_are_available(self):
        parent = node(
            "Assets",
            100,
            children=[
                node("Cash", 60, "ADD"),
                node("Receivables", 40, "ADD"),
                node("Displayed subtotal", 100, "TOTAL"),
            ],
        )
        report = reconcile_document(document(parent))
        self.assertEqual(report.reconciliations[0].expected, Decimal(100))
        self.assertEqual(report.inconsistencies, [])

    def test_total_only_children_are_included_as_subtotals(self):
        parent = node(
            "Assets",
            100,
            children=[
                node("Current assets", 70, "TOTAL"),
                node("Fixed assets", 30, "TOTAL"),
            ],
        )
        report = reconcile_document(document(parent))
        self.assertEqual(report.reconciliations[0].expected, Decimal(100))

    def test_skips_calculation_if_child_amount_is_unavailable(self):
        parent = node(
            "Assets",
            100,
            children=[
                node("Cash", 60, "ADD"),
                node("Receivables", None, "ADD", status="unreadable"),
            ],
        )
        report = reconcile_document(document(parent))
        self.assertEqual(report.reconciliations, [])
        self.assertEqual(len(report.skipped), 1)

    def test_tolerance_allows_small_difference(self):
        parent = node(
            "Assets",
            101,
            children=[node("Cash", 100, "ADD")],
        )
        report = reconcile_document(document(parent), Decimal(1))
        self.assertEqual(report.inconsistencies, [])

    def test_rejects_unknown_operator(self):
        parent = node("Assets", 100, children=[node("Cash", 100, "MULTIPLY")])
        with self.assertRaisesRegex(BalanceSheetDataError, "operator"):
            reconcile_document(document(parent))

    def test_nested_reported_category_is_reconciled_recursively(self):
        parent = node(
            "Assets",
            150,
            children=[
                node(
                    "Current assets",
                    100,
                    "ADD",
                    children=[
                        node("Cash", 60, "ADD"),
                        node("Receivables", 40, "ADD"),
                    ],
                ),
                node("Fixed assets", 50, "ADD"),
            ],
        )
        report = reconcile_document(document(parent))
        self.assertEqual(len(report.reconciliations), 2)
        self.assertEqual(report.inconsistencies, [])

    def test_cli_reports_mismatch_and_returns_nonzero(self):
        data = document(
            node(
                "Assets",
                100,
                children=[
                    node("Cash", 65, "ADD"),
                    node("Receivables", 30, "ADD"),
                ],
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "balance-sheet.json"
            json_path.write_text(json.dumps(data), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main([str(json_path)])

        self.assertEqual(result, 1)
        self.assertIn("MISMATCH", output.getvalue())
        self.assertIn("difference=5", output.getvalue())


if __name__ == "__main__":
    unittest.main()
