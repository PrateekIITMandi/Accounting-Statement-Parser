#!/usr/bin/env python3
"""Parse a Japanese balance-sheet PDF into the project's normalized JSON schema."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from reconcile_balance_sheet import print_report, reconcile_document


PROMPT = """Parse this Japanese corporate balance sheet and return one JSON object
conforming to the provided balance-sheet JSON Schema.

Use the PDF's visual layout, not OCR text order alone:
- Preserve filing order and use indentation, alignment, font boldness, and wrapped
  lines to infer grouping and distinguish headings, details, subtotals, and totals.
- Before assigning a visually indented row to a parent, verify that the
  relationship is semantically plausible in accounting context. Do not attach a
  row to a heading merely because it is nearby or indented.
- Join wrapped labels into one label. Do not treat a continuation line as a
  separate accounting item.
- Give each node a unique nodeId; represent hierarchy through children.
- Keep each Japanese label verbatim (apart from joining wrapped lines) in
  labels.ja, and translate each accounting label faithfully into English in
  labels.en. Do not select or map labels to an external taxonomy.
- Do not derive canonicalKey or aliases; Python creates the canonical key from
  the English translation.
- Use ISO dates. Detect the statement currency and printed unit header; for
  Japanese yen statements set currency to JPY and multiplier to 1 for 円,
  10000 for 万円, 1000 for 千円, or 1000000 for 百万円.
- For each value, preserve the printed token exactly in sourceText. Set amount
  to the signed numeric amount as printed, before applying the unit multiplier.
  Convert a printed negative marker such as △ or Δ to a negative number. A
  negative zero token such as △0 or ΔΟ is numeric zero. A dash is not zero:
  use amount null and an appropriate non-reported status.
- Tag each node's operator relative to its nearest applicable parent total:
  ADD for ordinary contributing items, SUBTRACT for contra-accounts (including
  減価償却累計額 and 貸倒引当金) even if the printed amount is positive, and
  TOTAL for subtotals, totals, and net/result rows such as 純額.
- Use an empty values array for headings/categories without an amount. For a
  reported zero, use status reported and amount 0.

Do not calculate totals, change the filing's reported values, or omit rows.
Return JSON only. The Python caller will preserve sourceText, normalize signs,
apply the multiplier to amounts, and validate the final object.
"""

_DASHES = frozenset({"-", "－", "–", "—", "―", "−", "ｰ", "ー"})
_ZERO_GLYPHS = frozenset({"0", "O", "Ο", "〇", "○"})
_AMOUNT_PATTERN = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)\Z")
_KNOWN_MULTIPLIERS = (
    ("百万円", 1_000_000),
    ("千円", 1_000),
    ("万円", 10_000),
    ("円", 1),
)


def parse_displayed_amount(source_text: str) -> Decimal | None:
    """Parse a printed amount; return None for blank/dash tokens."""
    token = "".join(source_text.split()).replace(",", "").replace("，", "")
    if not token or all(character in _DASHES for character in token):
        return None

    negative = False
    if token.startswith(("△", "Δ")):
        negative = True
        token = token[1:]
    elif token.startswith(("−", "-", "－")):
        negative = True
        token = token[1:]

    if len(token) >= 2 and token.startswith("(") and token.endswith(")"):
        negative = True
        token = token[1:-1]

    if token and all(character in _ZERO_GLYPHS for character in token):
        return Decimal(0)
    if not _AMOUNT_PATTERN.fullmatch(token):
        raise ValueError(f"Unrecognized amount token: {source_text!r}")

    try:
        amount = Decimal(token)
    except InvalidOperation as error:
        raise ValueError(f"Unrecognized amount token: {source_text!r}") from error
    return -amount if negative else amount


def _json_number(value: Decimal) -> int | float:
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _infer_multiplier(label_ja: str, supplied: Any) -> int:
    for unit_text, multiplier in _KNOWN_MULTIPLIERS:
        if unit_text in label_ja:
            return multiplier
    if isinstance(supplied, int) and not isinstance(supplied, bool) and supplied > 0:
        return supplied
    raise ValueError(
        f"Cannot determine a positive JPY multiplier from unit label {label_ja!r}"
    )


def _normalize_node(node: dict[str, Any], multiplier: int) -> None:
    for value in node["values"]:
        status = value["status"]
        if status == "reported":
            displayed_amount = parse_displayed_amount(value["sourceText"])
            if displayed_amount is None:
                raise ValueError(
                    f"Reported value {value['sourceText']!r} has no numeric amount"
                )
            value["amount"] = _json_number(displayed_amount * multiplier)
        else:
            value["amount"] = None

    for child in node["children"]:
        _normalize_node(child, multiplier)


def normalize_document(data: dict[str, Any], pdf_path: Path) -> dict[str, Any]:
    """Apply deterministic metadata and amount cleanup to the model response."""
    statement = data["statement"]
    unit = statement["unit"]
    unit["currency"] = "JPY"
    unit["multiplier"] = _infer_multiplier(unit["labelJa"], unit.get("multiplier"))
    data["sourceDocument"]["fileName"] = pdf_path.name

    for node in statement["items"]:
        _normalize_node(node, unit["multiplier"])
        _set_search_keys(node)
    return data


def _canonical_key(label: str) -> str:
    tokens: list[str] = []
    current: list[str] = []
    for character in unicodedata.normalize("NFKC", label):
        if character.isascii() and character.isalnum():
            current.append(character.upper())
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return "_".join(tokens) or "UNNAMED"


def _set_search_keys(node: dict[str, Any]) -> None:
    labels = node.get("labels")
    children = node.get("children", [])
    if not isinstance(labels, dict) or not isinstance(labels.get("en"), str):
        for child in children:
            _set_search_keys(child)
        return
    node["canonicalKey"] = _canonical_key(labels["en"])
    aliases = node.get("aliases", [])
    if not isinstance(aliases, list) or any(
        not isinstance(alias, str) or not alias.strip() for alias in aliases
    ):
        raise ValueError(f"Node {node.get('nodeId')!r} has invalid aliases")
    node["aliases"] = list(dict.fromkeys(alias.strip() for alias in aliases))
    for child in children:
        _set_search_keys(child)


def _normalize_search_term(term: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", term).casefold()
        if character.isalnum()
    )


class FinancialTableSearchEngine:
    """Exact-match O(1)-average lookup over a parsed statement."""

    def __init__(self, source: dict[str, Any] | str | Path):
        if isinstance(source, (str, Path)):
            try:
                document = json.loads(Path(source).read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid balance-sheet JSON: {error}") from error
        else:
            document = source

        statement = document.get("statement") if isinstance(document, dict) else None
        roots = statement.get("items") if isinstance(statement, dict) else None
        if not isinstance(roots, list):
            raise ValueError("Search index source must contain statement.items")
        self._index: dict[str, list[dict[str, Any]]] = {}
        self._node_ids: set[str] = set()
        for root in roots:
            self._index_node(root, "")

    def _index_node(self, node: Any, parent_path: str) -> None:
        if not isinstance(node, dict):
            raise ValueError("Every accounting item must be an object")
        node_id = node.get("nodeId")
        labels = node.get("labels")
        if not isinstance(node_id, str) or not isinstance(labels, dict):
            raise ValueError("Each item requires nodeId and bilingual labels")
        if node_id in self._node_ids:
            raise ValueError(f"Duplicate nodeId in search index: {node_id!r}")
        self._node_ids.add(node_id)
        japanese, english = labels.get("ja"), labels.get("en")
        if not isinstance(japanese, str) or not isinstance(english, str):
            raise ValueError(f"Node {node_id!r} requires Japanese and English labels")

        canonical_key = node.get("canonicalKey") or _canonical_key(english)
        aliases = node.get("aliases", [])
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or not alias.strip() for alias in aliases
        ):
            raise ValueError(f"Node {node_id!r} has invalid aliases")
        path = f"{parent_path} > {english} ({japanese})".strip(" >")
        result = {
            "nodeId": node_id,
            "canonicalKey": canonical_key,
            "labels": labels,
            "values": node.get("values", []),
            "path": path,
        }
        terms = {japanese, english, canonical_key, node_id, *aliases}
        for term in terms:
            normalized = _normalize_search_term(term)
            if not normalized:
                continue
            matches = self._index.setdefault(normalized, [])
            if all(match["nodeId"] != node_id for match in matches):
                matches.append(result)

        children = node.get("children")
        if not isinstance(children, list):
            raise ValueError(f"Node {node_id!r} children must be an array")
        for child in children:
            self._index_node(child, path)

    def search(self, query: str) -> list[dict[str, Any]]:
        """Return all exact normalized matches for an English or Japanese term."""
        normalized = _normalize_search_term(query)
        return list(self._index.get(normalized, ())) if normalized else []

    def get_value(self, query: str, period_id: str) -> int | float | None:
        """Return a unique item's amount for a given period."""
        matches = self.search(query)
        if not matches:
            raise KeyError(f"No accounting item matches {query!r}")
        if len(matches) > 1:
            ids = ", ".join(item["nodeId"] for item in matches)
            raise ValueError(f"Ambiguous search term {query!r}; matched: {ids}")
        for value in matches[0]["values"]:
            if value.get("periodId") == period_id:
                return value.get("amount")
        raise KeyError(
            f"No value for period {period_id!r} on {matches[0]['nodeId']!r}"
        )


def validate_document(data: dict[str, Any], schema_path: Path) -> None:
    """Validate schema constraints plus IDs and period references."""
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency: install requirements with "
            "`python -m pip install -r requirements.txt`."
        ) from error

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(data),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors[:10]
        )
        raise ValueError(f"Model output does not match the JSON Schema: {details}")

    statement = data["statement"]
    period_ids = [period["id"] for period in statement["periods"]]
    if len(period_ids) != len(set(period_ids)):
        raise ValueError("Statement contains duplicate period IDs")
    known_period_ids = set(period_ids)
    node_ids: set[str] = set()

    def inspect_node(node: dict[str, Any]) -> None:
        if node["nodeId"] in node_ids:
            raise ValueError(f"Statement contains duplicate nodeId {node['nodeId']!r}")
        node_ids.add(node["nodeId"])
        value_periods: set[str] = set()
        for value in node["values"]:
            period_id = value["periodId"]
            if period_id not in known_period_ids:
                raise ValueError(
                    f"Value references unknown periodId {period_id!r}"
                )
            if period_id in value_periods:
                raise ValueError(
                    f"Node {node['nodeId']!r} contains duplicate values for "
                    f"period {period_id!r}"
                )
            value_periods.add(period_id)
        for child in node["children"]:
            inspect_node(child)

    for item in statement["items"]:
        inspect_node(item)


def _gemini_sdk() -> tuple[Any, Any, Any, str]:
    try:
        from google import genai
        from google.genai import errors, types
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency: install requirements with "
            "`python -m pip install -r requirements.txt`."
        ) from error

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY before running the parser."
        )
    return genai, errors, types, api_key


def _generate_json(model: str, contents: list[Any]) -> dict[str, Any]:
    genai, errors, types, api_key = _gemini_sdk()
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
    except errors.APIError as error:
        raise RuntimeError(f"Gemini request failed: {error}") from error
    finally:
        client.close()
    if not isinstance(response.text, str) or not response.text:
        raise ValueError("Gemini returned no JSON content")
    try:
        result = json.loads(response.text)
    except json.JSONDecodeError as error:
        raise ValueError("Gemini did not return valid JSON") from error
    if not isinstance(result, dict):
        raise ValueError("Gemini response must be a JSON object")
    return result


def request_model_json(
    pdf_path: Path,
    model: str,
    schema_path: Path,
) -> dict[str, Any]:
    """Extract the PDF and translate Japanese terms without taxonomy matching."""
    _, _, types, _ = _gemini_sdk()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["$defs"]["reportedValue"]["properties"]["amount"]["description"] = (
        "For this model response only: signed numeric amount as printed, before "
        "applying statement.unit.multiplier. The Python normalizer scales it to "
        "yen before final schema validation."
    )
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    prompt = (
        f"{PROMPT}\n\n"
        f"JSON SCHEMA:\n{schema_text}"
    )
    return _generate_json(
        model,
        [
            types.Part.from_bytes(
                data=pdf_path.read_bytes(),
                mime_type="application/pdf",
            ),
            prompt,
        ],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a Japanese balance-sheet PDF, validate its JSON, reconcile "
            "the totals, and provide exact-term search."
        )
    )
    parser.add_argument("pdf", type=Path, help="Input balance-sheet PDF")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON path (default: input PDF path with .json extension)",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).with_name("balance-sheet.schema.json"),
        help="JSON Schema path (default: balance-sheet.schema.json beside this script)",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.6-flash",
        help="Google Gemini PDF-capable vision model (default: gemini-3.6-flash)",
    )
    parser.add_argument(
        "--tolerance",
        type=Decimal,
        default=None,
        help=(
            "Explicit allowed reconciliation difference in yen; by default "
            "tolerance is derived from the statement's displayed unit"
        ),
    )
    parser.add_argument(
        "--search",
        action="append",
        default=[],
        metavar="TERM",
        help="Search an exact Japanese/English label, canonical key, or alias",
    )
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> tuple[Path, Any]:
    if args.tolerance is not None and (
        not args.tolerance.is_finite() or args.tolerance < 0
    ):
        raise ValueError("--tolerance must be a finite, non-negative number")
    pdf_path = args.pdf.expanduser().resolve()
    schema_path = args.schema.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else pdf_path.with_suffix(".json")
    )
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")
    if not schema_path.is_file():
        raise FileNotFoundError(f"JSON Schema not found: {schema_path}")

    protected_paths = {pdf_path, schema_path}
    if output_path in protected_paths:
        raise ValueError("Output JSON path must not overwrite an input or schema file")

    extracted = request_model_json(pdf_path, args.model, schema_path)
    normalized = normalize_document(extracted, pdf_path)
    validate_document(normalized, schema_path)
    output_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    saved_document = json.loads(output_path.read_text(encoding="utf-8"))
    validate_document(saved_document, schema_path)
    search_engine = FinancialTableSearchEngine(saved_document)
    for query in args.search:
        matches = search_engine.search(query)
        if not matches:
            print(f"SEARCH: no exact match for {query!r}", file=sys.stderr)
            continue
        for match in matches:
            print(
                f"SEARCH: {query!r} -> {match['canonicalKey']} "
                f"{match['labels']['en']} ({match['labels']['ja']}); "
                f"values={json.dumps(match['values'], ensure_ascii=False)}",
                file=sys.stderr,
            )
    report = reconcile_document(saved_document, args.tolerance)
    return output_path, report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_path, report = run_pipeline(args)
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Wrote validated balance sheet JSON to {output_path}", file=sys.stderr)
    print_report(report, args.tolerance, stream=sys.stderr)
    if not report.reconciliations:
        return 2
    return 1 if report.inconsistencies else 0


if __name__ == "__main__":
    raise SystemExit(main())
