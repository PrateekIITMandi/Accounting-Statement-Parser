import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from parse_balance_sheet import (
    FinancialTableSearchEngine,
    normalize_document,
    parse_args,
    parse_displayed_amount,
    main,
    request_model_json,
    run_pipeline,
    validate_document,
)
from reconcile_balance_sheet import ReconciliationReport


class ParseDisplayedAmountTests(unittest.TestCase):
    def test_strips_internal_spaces_and_grouping(self):
        self.assertEqual(parse_displayed_amount("1, 923, 585"), 1923585)

    def test_parses_japanese_negative_marker(self):
        self.assertEqual(parse_displayed_amount("△2,979"), -2979)

    def test_normalizes_negative_zero_ocr_variants(self):
        self.assertEqual(parse_displayed_amount("△0"), 0)
        self.assertEqual(parse_displayed_amount("ΔΟ"), 0)

    def test_dash_is_not_zero(self):
        self.assertIsNone(parse_displayed_amount("－"))

    def test_rejects_unrecognized_text(self):
        with self.assertRaises(ValueError):
            parse_displayed_amount("not an amount")


class NormalizeDocumentTests(unittest.TestCase):
    def test_scales_to_yen_and_preserves_source_text(self):
        document = {
            "sourceDocument": {
                "fileName": "model-name.pdf",
                "sourceLanguage": "ja",
                "correctionStatus": "corrected",
            },
            "statement": {
                "unit": {"currency": "JPY", "multiplier": 1, "labelJa": "千円"},
                "items": [
                    {
                        "values": [
                            {
                                "status": "reported",
                                "sourceText": "1, 923, 585",
                                "amount": 1923585,
                            },
                            {
                                "status": "reported",
                                "sourceText": "ΔΟ",
                                "amount": 0,
                            },
                        ],
                        "children": [],
                    }
                ],
            },
        }

        normalized = normalize_document(document, Path("sample.pdf"))
        values = normalized["statement"]["items"][0]["values"]
        self.assertEqual(normalized["sourceDocument"]["fileName"], "sample.pdf")
        self.assertEqual(normalized["statement"]["unit"]["multiplier"], 1000)
        self.assertEqual(values[0]["amount"], 1923585000)
        self.assertEqual(values[0]["sourceText"], "1, 923, 585")
        self.assertEqual(values[1]["amount"], 0)

    def test_dash_status_is_null(self):
        document = {
            "sourceDocument": {
                "fileName": "model-name.pdf",
                "sourceLanguage": "ja",
                "correctionStatus": "corrected",
            },
            "statement": {
                "unit": {"currency": "JPY", "multiplier": 1000, "labelJa": "千円"},
                "items": [
                    {
                        "values": [
                            {
                                "status": "not_applicable",
                                "sourceText": "－",
                                "amount": None,
                            }
                        ],
                        "children": [],
                    }
                ],
            },
        }

        normalized = normalize_document(document, Path("sample.pdf"))
        self.assertIsNone(
            normalized["statement"]["items"][0]["values"][0]["amount"]
        )


class SchemaValidationTests(unittest.TestCase):
    def setUp(self):
        labels = {"ja": "資産合計", "en": "Total assets"}
        self.document = {
            "sourceDocument": {
                "fileName": "sample.pdf",
                "sourceLanguage": "ja",
                "correctionStatus": "corrected",
            },
            "statement": {
                "statementType": "consolidated_balance_sheet",
                "labels": {"ja": "連結貸借対照表", "en": "Consolidated balance sheet"},
                "periods": [
                    {
                        "id": "current",
                        "kind": "current",
                        "date": "2026-03-31",
                        "labels": {"ja": "当連結会計年度", "en": "Current fiscal year"},
                    }
                ],
                "unit": {"currency": "JPY", "multiplier": 1000, "labelJa": "千円"},
                "items": [
                    {
                        "nodeId": "assets",
                        "kind": "total",
                        "labels": labels,
                        "operator": "TOTAL",
                        "values": [
                            {
                                "periodId": "current",
                                "status": "reported",
                                "amount": 0,
                                "sourceText": "△0",
                            }
                        ],
                        "children": [],
                    }
                ],
            },
        }
        self.schema_path = Path(__file__).with_name("balance-sheet.schema.json")

    def test_accepts_schema_conformant_normalized_document(self):
        validate_document(self.document, self.schema_path)

    def test_rejects_unknown_period_reference(self):
        value = self.document["statement"]["items"][0]["values"][0]
        value["periodId"] = "missing"
        with self.assertRaisesRegex(ValueError, "unknown periodId"):
            validate_document(self.document, self.schema_path)


class FinancialTableSearchEngineTests(unittest.TestCase):
    def setUp(self):
        self.document = {
            "statement": {
                "items": [
                    {
                        "nodeId": "cash",
                        "canonicalKey": "CASH_AND_DEPOSITS",
                        "aliases": ["Cash equivalents"],
                        "labels": {
                            "ja": "現金及び預金",
                            "en": "Cash and deposits",
                        },
                        "values": [
                            {
                                "periodId": "current",
                                "status": "reported",
                                "amount": 608281000,
                                "sourceText": "608,281",
                            }
                        ],
                        "children": [],
                    }
                ]
            }
        }
        self.engine = FinancialTableSearchEngine(self.document)

    def test_japanese_english_canonical_and_alias_search(self):
        for term in (
            "現金及び預金",
            "Cash and deposits",
            "CASH_AND_DEPOSITS",
            "Cash equivalents",
        ):
            with self.subTest(term=term):
                self.assertEqual(self.engine.search(term)[0]["nodeId"], "cash")

    def test_search_accepts_normalized_spacing_and_case(self):
        self.assertEqual(
            self.engine.search("cash and deposits"),
            self.engine.search("CASH_AND_DEPOSITS"),
        )

    def test_get_value_returns_amount_for_period(self):
        self.assertEqual(
            self.engine.get_value("現金及び預金", "current"),
            608281000,
        )

    def test_cli_accepts_search_option(self):
        args = parse_args(
            [
                "balance_sheet_after_correction.pdf",
                "--search",
                "現金及び預金",
            ]
        )
        self.assertEqual(args.search, ["現金及び預金"])

    def test_cli_defaults_to_gemini_36_flash(self):
        args = parse_args(["balance_sheet_after_correction.pdf"])
        self.assertEqual(args.model, "gemini-3.6-flash")


class GeminiRequestTests(unittest.TestCase):
    def test_sends_pdf_to_gemini_and_parses_json_response(self):
        response = SimpleNamespace(text='{"extracted": true}')
        client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=Mock(return_value=response)
            ),
            close=Mock(),
        )
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pdf_path = root / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 sample")
            schema_path = Path(__file__).with_name("balance-sheet.schema.json")
            with patch("google.genai.Client", return_value=client) as client_factory:
                with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
                    result = request_model_json(
                        pdf_path,
                        "gemini-3.6-flash",
                        schema_path,
                    )

        self.assertEqual(result, {"extracted": True})
        client_factory.assert_called_once_with(api_key="test-key")
        call = client.models.generate_content.call_args.kwargs
        self.assertEqual(call["model"], "gemini-3.6-flash")
        self.assertEqual(call["contents"][0].inline_data.mime_type, "application/pdf")
        self.assertEqual(call["config"].response_mime_type, "application/json")
        client.close.assert_called_once()


class PipelineTests(unittest.TestCase):
    def test_pipeline_saves_json_and_reconciles_saved_output(self):
        cash = {
            "nodeId": "cash",
            "kind": "line_item",
            "labels": {"ja": "現金及び預金", "en": "Cash and deposits"},
            "operator": "ADD",
            "values": [
                {
                    "periodId": "current",
                    "status": "reported",
                    "amount": 700,
                    "sourceText": "700",
                }
            ],
            "children": [],
        }
        receivables = {
            **cash,
            "nodeId": "receivables",
            "labels": {"ja": "売掛金", "en": "Accounts receivable"},
            "values": [
                {
                    "periodId": "current",
                    "status": "reported",
                    "amount": 300,
                    "sourceText": "300",
                }
            ],
        }
        model_document = {
            "sourceDocument": {
                "fileName": "model.pdf",
                "sourceLanguage": "ja",
                "correctionStatus": "corrected",
            },
            "statement": {
                "statementType": "consolidated_balance_sheet",
                "labels": {
                    "ja": "連結貸借対照表",
                    "en": "Consolidated balance sheet",
                },
                "periods": [
                    {
                        "id": "current",
                        "kind": "current",
                        "date": "2026-03-31",
                        "labels": {
                            "ja": "当連結会計年度",
                            "en": "Current fiscal year",
                        },
                    }
                ],
                "unit": {"currency": "JPY", "multiplier": 1000, "labelJa": "千円"},
                "items": [
                    {
                        "nodeId": "assets",
                        "kind": "total",
                        "labels": {"ja": "資産合計", "en": "Total assets"},
                        "operator": "TOTAL",
                        "values": [
                            {
                                "periodId": "current",
                                "status": "reported",
                                "amount": 1000,
                                "sourceText": "1,000",
                            }
                        ],
                        "children": [cash, receivables],
                    }
                ],
            },
        }

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pdf_path = root / "sample.pdf"
            pdf_path.write_bytes(b"sample input")
            output_path = root / "sample.json"
            args = SimpleNamespace(
                pdf=pdf_path,
                schema=Path(__file__).with_name("balance-sheet.schema.json"),
                output=output_path,
                model="gemini-3.6-flash",
                tolerance=Decimal(0),
                search=["CASH_AND_DEPOSITS"],
            )
            with patch(
                "parse_balance_sheet.request_model_json",
                return_value=model_document,
            ):
                saved_path, report = run_pipeline(args)

            saved = json.loads(saved_path.read_text(encoding="utf-8"))

        self.assertEqual(saved_path, output_path)
        self.assertEqual(
            saved["statement"]["items"][0]["children"][0]["values"][0]["amount"],
            700000,
        )
        self.assertEqual(
            saved["statement"]["items"][0]["children"][0]["labels"]["en"],
            "Cash and deposits",
        )
        self.assertNotIn(
            "taxonomyMatch", saved["statement"]["items"][0]["children"][0]
        )
        self.assertNotIn(
            "translatedEn", saved["statement"]["items"][0]["children"][0]
        )
        self.assertEqual(report.inconsistencies, [])

    def test_cli_does_not_print_json_to_stdout(self):
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "sample.json"
            output_path.write_text('{"saved": true}\n', encoding="utf-8")
            stdout = StringIO()
            stderr = StringIO()
            with patch(
                "parse_balance_sheet.run_pipeline",
                return_value=(output_path, ReconciliationReport([], [])),
            ):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    result = main(["sample.pdf"])

        self.assertEqual(result, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn(str(output_path), stderr.getvalue())
        self.assertIn("arithmetic consistency was not verified", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
