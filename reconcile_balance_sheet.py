#!/usr/bin/env python3
"""Check arithmetic consistency in a parsed balance-sheet JSON document."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, TextIO


_OPERATORS = {
    "ADD": "ADD",
    "+": "ADD",
    "SUBTRACT": "SUBTRACT",
    "-": "SUBTRACT",
    "−": "SUBTRACT",
    "TOTAL": "TOTAL",
}


@dataclass(frozen=True)
class Reconciliation:
    path: str
    node_id: str
    period_id: str
    period_label: str
    expected: Decimal
    actual: Decimal
    difference: Decimal
    allowed_difference: Decimal


@dataclass(frozen=True)
class SkippedCheck:
    path: str
    period_id: str
    reason: str


@dataclass
class ReconciliationReport:
    reconciliations: list[Reconciliation]
    skipped: list[SkippedCheck]

    @property
    def inconsistencies(self) -> list[Reconciliation]:
        return [
            item
            for item in self.reconciliations
            if abs(item.difference) > item.allowed_difference
        ]


class BalanceSheetDataError(ValueError):
    """Raised when the input does not have usable balance-sheet JSON structure."""


def _decimal_amount(value: Any, context: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise BalanceSheetDataError(f"{context}: amount must be a JSON number")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as error:
        raise BalanceSheetDataError(f"{context}: invalid numeric amount") from error
    if not amount.is_finite():
        raise BalanceSheetDataError(f"{context}: amount must be finite")
    return amount


def _operator(node: dict[str, Any], context: str) -> str:
    raw_operator = node.get("operator")
    if not isinstance(raw_operator, str) or raw_operator.upper() not in _OPERATORS:
        raise BalanceSheetDataError(
            f"{context}: operator must be ADD, SUBTRACT, TOTAL, +, or -"
        )
    return _OPERATORS[raw_operator.upper()]


def _label(node: dict[str, Any]) -> str:
    labels = node.get("labels")
    if isinstance(labels, dict):
        english = labels.get("en")
        japanese = labels.get("ja")
        if isinstance(english, str) and isinstance(japanese, str):
            return f"{english} ({japanese})"
        if isinstance(english, str):
            return english
        if isinstance(japanese, str):
            return japanese
    return str(node.get("nodeId", "<unnamed>"))


def reconcile_document(
    document: dict[str, Any], tolerance: Decimal | None = None
) -> ReconciliationReport:
    """Reconcile both nested totals and sibling subtotal rows.

    Japanese balance sheets commonly place detail rows and their total as
    siblings in a category's ``children`` array. Net rows are matched to their
    corresponding gross and contra-account using canonical keys, then treated
    as one contribution in higher-level totals.
    """
    if tolerance is not None and (not tolerance.is_finite() or tolerance < 0):
        raise ValueError("tolerance must be finite and non-negative")
    if not isinstance(document, dict):
        raise BalanceSheetDataError("JSON root must be an object")
    statement = document.get("statement")
    if not isinstance(statement, dict):
        raise BalanceSheetDataError("JSON must contain a statement object")

    periods_data = statement.get("periods")
    roots = statement.get("items")
    if not isinstance(periods_data, list) or not isinstance(roots, list):
        raise BalanceSheetDataError(
            "statement.periods and statement.items must be arrays"
        )
    unit = statement.get("unit", {})
    multiplier = unit.get("multiplier", 1) if isinstance(unit, dict) else 1
    if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier < 1:
        raise BalanceSheetDataError("statement.unit.multiplier must be a positive integer")
    periods: dict[str, str] = {}
    for index, period in enumerate(periods_data):
        if not isinstance(period, dict) or not isinstance(period.get("id"), str):
            raise BalanceSheetDataError(
                f"statement.periods[{index}] must have a string id"
            )
        period_id = period["id"]
        if period_id in periods:
            raise BalanceSheetDataError(f"duplicate period id {period_id!r}")
        labels = period.get("labels")
        label = labels.get("en") if isinstance(labels, dict) else None
        if not isinstance(label, str):
            label = period_id
        periods[period_id] = label

    checks: list[Reconciliation] = []
    skipped: list[SkippedCheck] = []
    node_ids: set[str] = set()

    def index_values(node: dict[str, Any], path: str) -> dict[str, Decimal | None]:
        values = node.get("values")
        if not isinstance(values, list):
            raise BalanceSheetDataError(f"{path}: values must be an array")
        indexed: dict[str, Decimal | None] = {}
        for value_index, value in enumerate(values):
            value_path = f"{path}.values[{value_index}]"
            if not isinstance(value, dict):
                raise BalanceSheetDataError(f"{value_path}: expected an object")
            period_id = value.get("periodId")
            if not isinstance(period_id, str) or period_id not in periods:
                raise BalanceSheetDataError(
                    f"{value_path}: unknown or missing periodId"
                )
            if period_id in indexed:
                raise BalanceSheetDataError(
                    f"{path}: duplicate value for period {period_id!r}"
                )
            status = value.get("status")
            amount = value.get("amount")
            if status == "reported":
                indexed[period_id] = _decimal_amount(amount, value_path)
            elif status in {"not_reported", "not_applicable", "unreadable"} and amount is None:
                source_text = value.get("sourceText", "")
                if status == "not_applicable" or (
                    status == "not_reported"
                    and isinstance(source_text, str)
                    and source_text.strip() in {"-", "－", "–", "—", "―", "△0", "Δ0", "ΔΟ"}
                ):
                    indexed[period_id] = Decimal(0)
                else:
                    indexed[period_id] = None
            else:
                raise BalanceSheetDataError(
                    f"{value_path}: status and amount are inconsistent"
                )
        return indexed

    def net_base_key(node: dict[str, Any]) -> str | None:
        canonical_key = node.get("canonicalKey")
        if isinstance(canonical_key, str) and canonical_key.endswith("_NET"):
            return canonical_key[:-4]
        labels = node.get("labels")
        english = labels.get("en") if isinstance(labels, dict) else None
        if isinstance(english, str) and english.casefold().endswith(", net"):
            return english[:-5].strip().casefold()
        japanese = labels.get("ja") if isinstance(labels, dict) else None
        if isinstance(japanese, str) and "純額" in japanese:
            return japanese.replace("（純額）", "").replace("(純額)", "").strip()
        return None

    def is_net_total(node: dict[str, Any]) -> bool:
        return net_base_key(node) is not None

    def is_aggregate_total(node: dict[str, Any]) -> bool:
        if node.get("kind") == "total":
            return True
        labels = node.get("labels")
        english = labels.get("en") if isinstance(labels, dict) else None
        japanese = labels.get("ja") if isinstance(labels, dict) else None
        return (
            isinstance(english, str)
            and english.casefold().startswith("total ")
        ) or (isinstance(japanese, str) and japanese.endswith("合計"))

    def find_net_components(
        target: dict[str, Any], previous: list[dict[str, Any]]
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        base_key = net_base_key(target)
        if base_key is None:
            return None, None
        gross = [
            item
            for item in previous
            if _operator(item, _label(item)) == "ADD"
            and item.get("canonicalKey") == base_key
        ]
        contra = [
            item
            for item in previous
            if _operator(item, _label(item)) == "SUBTRACT"
            and isinstance(item.get("canonicalKey"), str)
            and base_key in item["canonicalKey"]
            and (
                "DEPRECIATION" in item["canonicalKey"]
                or "ALLOWANCE" in item["canonicalKey"]
            )
        ]
        if len(gross) == 1 and len(contra) == 1:
            return gross[0], contra[0]
        return None, None

    def inspect_node(
        node: Any, parent_path: str
    ) -> tuple[dict[str, Decimal | None], list[dict[str, Any]]]:
        if not isinstance(node, dict):
            raise BalanceSheetDataError(f"{parent_path}: item must be an object")
        node_id = node.get("nodeId")
        if not isinstance(node_id, str) or not node_id:
            raise BalanceSheetDataError(f"{parent_path}: nodeId must be a non-empty string")
        if node_id in node_ids:
            raise BalanceSheetDataError(f"duplicate nodeId {node_id!r}")
        node_ids.add(node_id)
        path = f"{parent_path} > {_label(node)}" if parent_path else _label(node)
        _operator(node, path)
        own_values = index_values(node, path)
        children = node.get("children")
        if not isinstance(children, list):
            raise BalanceSheetDataError(f"{path}: children must be an array")

        child_values = reconcile_siblings(children, path)
        effective_values = dict(own_values)
        if children and any(amount is not None for amount in own_values.values()):
            contributors = select_contributors(node, children)
            if contributors is None:
                for period_id in periods:
                    if own_values.get(period_id) is not None:
                        skipped.append(
                            SkippedCheck(
                                path,
                                period_id,
                                "could not uniquely match net row to a gross item and contra-account",
                            )
                        )
            else:
                record_total_check(
                    node,
                    own_values,
                    contributors,
                    path,
                    child_values,
                )
        if not any(amount is not None for amount in effective_values.values()):
            terminal_totals = [
                child
                for child in children
                if isinstance(child, dict)
                and _operator(child, path) == "TOTAL"
                and not is_net_total(child)
            ]
            if terminal_totals:
                effective_values.update(child_values.get(id(terminal_totals[-1]), {}))
        return effective_values, children

    def select_contributors(
        target: dict[str, Any], previous: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        if is_net_total(target):
            gross, contra = find_net_components(target, previous)
            if gross is None or contra is None:
                return None
            return [gross, contra]

        absorbed_ids: set[str] = set()
        for index, prior_total in enumerate(previous):
            if not is_net_total(prior_total):
                continue
            prior_gross, prior_contra = find_net_components(
                prior_total, previous[:index]
            )
            if prior_gross is not None and prior_contra is not None:
                absorbed_ids.update(
                    {prior_gross["nodeId"], prior_contra["nodeId"]}
                )
        has_signed_children = any(
            _operator(item, _label(item)) in {"ADD", "SUBTRACT"}
            for item in previous
        )
        return [
            item
            for item in previous
            if item.get("nodeId") not in absorbed_ids
            and not (
                has_signed_children
                and _operator(item, _label(item)) == "TOTAL"
                and not is_net_total(item)
            )
        ]

    def record_total_check(
        target: dict[str, Any],
        target_values: dict[str, Decimal | None],
        contributors: list[dict[str, Any]],
        target_path: str,
        effective_values: dict[int, dict[str, Decimal | None]],
    ) -> None:
        for period_id, period_label in periods.items():
            actual = target_values.get(period_id)
            if actual is None:
                continue
            expected = Decimal(0)
            missing: list[str] = []
            for item in contributors:
                item_amount = effective_values.get(id(item), {}).get(period_id)
                if item_amount is None:
                    missing.append(_label(item))
                    continue
                if _operator(item, target_path) == "SUBTRACT":
                    expected -= abs(item_amount)
                else:
                    expected += item_amount
            if missing:
                skipped.append(
                    SkippedCheck(
                        target_path,
                        period_id,
                        "missing reported amount for: " + ", ".join(missing),
                    )
                )
                continue
            difference = actual - expected
            allowed_difference = (
                tolerance
                if tolerance is not None
                else Decimal(multiplier) * Decimal(len(contributors) + 1) / 2
            )
            checks.append(
                Reconciliation(
                    path=target_path,
                    node_id=target["nodeId"],
                    period_id=period_id,
                    period_label=period_label,
                    expected=expected,
                    actual=actual,
                    difference=difference,
                    allowed_difference=allowed_difference,
                )
            )

    def reconcile_siblings(
        siblings: list[Any], parent_path: str
    ) -> dict[int, dict[str, Decimal | None]]:
        child_effective: dict[int, dict[str, Decimal | None]] = {}
        sibling_paths: dict[int, str] = {}
        for sibling in siblings:
            node_path = (
                f"{parent_path} > {_label(sibling)}"
                if parent_path
                else _label(sibling) if isinstance(sibling, dict) else parent_path
            )
            sibling_paths[id(sibling)] = node_path
            values, _ = inspect_node(sibling, parent_path)
            child_effective[id(sibling)] = values

        for index, target in enumerate(siblings):
            if (
                not isinstance(target, dict)
                or _operator(target, parent_path) != "TOTAL"
                or not is_aggregate_total(target) and not is_net_total(target)
            ):
                continue
            if index == 0:
                continue
            target_values = index_values(target, sibling_paths[id(target)])
            previous = [
                item for item in siblings[:index] if isinstance(item, dict)
            ]
            contributors = select_contributors(target, previous)
            if contributors is None:
                for period_id in periods:
                    if target_values.get(period_id) is not None:
                        skipped.append(
                            SkippedCheck(
                                sibling_paths[id(target)],
                                period_id,
                                "could not uniquely match net row to a gross item and contra-account",
                            )
                        )
                continue
            record_total_check(
                target,
                target_values,
                contributors,
                sibling_paths[id(target)],
                child_effective,
            )
        return child_effective

    reconcile_siblings(roots, "")
    return ReconciliationReport(checks, skipped)


def _format_amount(amount: Decimal) -> str:
    return format(amount, "f")


def print_report(
    report: ReconciliationReport,
    tolerance: Decimal | None,
    stream: TextIO | None = None,
) -> None:
    output = stream or sys.stdout
    if report.reconciliations:
        for result in report.reconciliations:
            if result.difference == 0:
                label = "OK"
            elif abs(result.difference) <= result.allowed_difference:
                label = "ROUNDING"
            else:
                label = "MISMATCH"
            print(
                f"{label}: {result.path} [{result.period_label}] "
                f"children={_format_amount(result.expected)} "
                f"reported={_format_amount(result.actual)} "
                f"difference={_format_amount(result.difference)} "
                f"allowed={_format_amount(result.allowed_difference)}",
                file=output,
            )
    else:
        print(
            "No reported totals with available child amounts were reconciled.",
            file=output,
        )

    if report.skipped:
        print(f"Skipped checks: {len(report.skipped)}", file=output)
        for skipped in report.skipped:
            print(
                f"SKIPPED: {skipped.path} [{skipped.period_id}] "
                f"{skipped.reason}",
                file=output,
            )

    mismatch_count = len(report.inconsistencies)
    if mismatch_count:
        print(
            f"Found {mismatch_count} inconsistency/inconsistencies in "
            f"{len(report.reconciliations)} reconciliation check(s) "
            f"(explicit tolerance: "
            f"{_format_amount(tolerance) if tolerance is not None else 'unit-based rounding'}).",
            file=output,
        )
    elif not report.reconciliations:
        print(
            "No reconciliation checks were performed; arithmetic consistency "
            "was not verified.",
            file=output,
        )
    else:
        rounded_count = sum(
            item.difference != 0 for item in report.reconciliations
        )
        print(
            f"Checked {len(report.reconciliations)} total(s); no arithmetic "
            f"inconsistencies found ({rounded_count} rounding differences).",
            file=output,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check parent totals against signed child items in balance-sheet JSON."
    )
    parser.add_argument("json_file", type=Path, help="Parsed balance-sheet JSON")
    parser.add_argument(
        "--tolerance",
        type=Decimal,
        default=None,
        help=(
            "Explicit allowed difference in stored currency units; by default "
            "tolerance is derived from the displayed unit and number of values"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tolerance is not None and (
        not args.tolerance.is_finite() or args.tolerance < 0
    ):
        print(
            "Error: --tolerance must be a finite, non-negative number",
            file=sys.stderr,
        )
        return 2
    json_path = args.json_file.expanduser().resolve()
    try:
        document = json.loads(json_path.read_text(encoding="utf-8"))
        report = reconcile_document(document, args.tolerance)
    except (OSError, json.JSONDecodeError, BalanceSheetDataError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print_report(report, args.tolerance)
    if not report.reconciliations:
        return 2
    return 1 if report.inconsistencies else 0


if __name__ == "__main__":
    raise SystemExit(main())
