# Japanese Balance Sheet Parser

This repository parses Japanese corporate balance-sheet PDFs into validated,
bilingual JSON. It uses Gemini 3.6 Flash to interpret the statement's visual
layout, translate accounting labels into English, and extract the reported
amounts. Python then normalizes the amounts, validates the JSON, and checks
reported totals for arithmetic consistency.

## Why hierarchical JSON?

Balance sheets are more than lists of labels and values: indentation and
subtotal rows express accounting relationships. A hierarchical representation
preserves those relationships with parent items and nested `children`, while
keeping each Japanese label alongside its English translation. It also retains
periods, source amount text, normalized values, and each row's `ADD`,
`SUBTRACT`, or `TOTAL` operator. This makes the extracted data easier to
validate, reconcile, search, and use downstream without flattening away its
accounting structure.

## Setup and run

Use Python 3.10 or later. From the repository directory, create and activate a
virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Set a Google Gemini API key in the environment. The parser accepts either
`GEMINI_API_KEY` or `GOOGLE_API_KEY`; do not commit the key to the repository.

```bash
export GEMINI_API_KEY="your-api-key"
```

On PowerShell:

```powershell
$env:GEMINI_API_KEY = "your-api-key"
```

Parse a PDF using the default model (`gemini-3.6-flash`):

```bash
python parse_balance_sheet.py balance_sheet_after_correction.pdf
```

By default, the JSON is saved beside the PDF with the same base name. To choose
an output path, select a model, search for a Japanese or English label, or set
an explicit reconciliation tolerance in yen:

```bash
python parse_balance_sheet.py input.pdf \
  --output output.json \
  --model gemini-3.6-flash \
  --search "現金及び預金" \
  --tolerance 1000
```

`--search` can be repeated. Without `--tolerance`, the reconciler derives an
allowance from the statement's displayed unit to account for rounding in
independently rounded displayed amounts. The parser saves the JSON rather than
printing the full document to the terminal; it reports the output path,
reconciliation results, and any requested search results.

The standalone reconciler can also check an existing JSON file:

```bash
python reconcile_balance_sheet.py output.json
```

## Extraction and normalization

1. The parser sends the PDF and the JSON Schema to Gemini 3.6 Flash with
   instructions to use indentation, alignment, font weight, and wrapped lines
   to recover the hierarchy. It asks the model to check whether a visually
   indented row belongs to its proposed parent semantically.
2. The model returns each Japanese label with its English translation, filing
   order, periods, displayed unit, values, and accounting operator. No EDINET
   taxonomy matching is performed.
3. Python preserves each printed amount in `sourceText`, normalizes signs and
   zero-like marks, applies the statement's unit multiplier to store values in
   yen, and generates canonical search keys from English labels.
4. The resulting document is validated against
   [`balance-sheet.schema.json`](balance-sheet.schema.json), saved, reloaded,
   and validated again before reconciliation.

The checked-in
[`balance_sheet_after_correction.json`](balance_sheet_after_correction.json)
is an example of the bilingual hierarchical output.

## Reconciliation engine

The reconciler checks whether a reported total agrees with its contributing
rows for each period. It handles nested totals and sibling-row layouts, applies
`ADD` and `SUBTRACT` operators, and excludes intermediate totals where detail
rows are available. For net rows, it checks the corresponding gross amount
minus its contra-account. Checks are skipped when required values are missing
or the gross/contra rows cannot be matched reliably.

Differences within the derived rounding allowance are reported as rounding
differences; larger differences are reported as inconsistencies. A report that
performs no checks is explicitly marked as unverified. The command exits with
status `0` when checks complete without inconsistencies, `1` when an
inconsistency is found, and `2` for an error or when no checks could verify the
document.
