#!/usr/bin/env python3
"""Data Quality Guardian: audit CSV data and optionally evaluate known injected errors."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "order_id", "order_date", "customer_id", "product_category",
    "quantity", "unit_price", "region"
]
APPROVED_CATEGORIES = {
    "Electronics", "Home Decor", "Office Supplies", "Clothing", "Groceries"
}
CATEGORY_ALIASES = {
    "home decor": "Home Decor",
    "homedecor": "Home Decor",
    "office supplies": "Office Supplies",
    "officesupplies": "Office Supplies",
    "electronics": "Electronics",
    "clothing": "Clothing",
    "groceries": "Groceries",
}
MISSING_MARKERS = {"", "na", "n/a", "null", "none", "nan", "missing"}
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def is_missing(value: Any) -> bool:
    """Treat actual nulls and common text placeholders as missing."""
    if pd.isna(value):
        return True
    return str(value).strip().lower() in MISSING_MARKERS


def add_issue(issues: list[dict[str, Any]], row_index: Any, row_number: int,
              column: str, rule: str, value: Any, message: str) -> None:
    issues.append({
        "row_index": int(row_index),
        "csv_row_number": int(row_number),
        "column": column,
        "rule": rule,
        "value": "" if is_missing(value) else str(value),
        "message": message,
    })


def audit_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Audit dataframe. csv_row_number is the physical CSV line (header is line 1)."""
    work = df.copy()
    issues: list[dict[str, Any]] = []

    missing_columns = [c for c in EXPECTED_COLUMNS if c not in work.columns]
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {', '.join(missing_columns)}")

    # Missing values and common text markers.
    for col in EXPECTED_COLUMNS:
        for idx, value in work[col].items():
            if is_missing(value):
                add_issue(issues, idx, int(idx) + 2, col, "missing_value", value,
                          f"{col} is blank or uses a recognized missing-value marker.")

    # Exact duplicate rows: mark later occurrences only.
    duplicated = work.duplicated(keep="first")
    for idx in work.index[duplicated]:
        add_issue(issues, idx, int(idx) + 2, "__row__", "duplicate_row", "",
                  "This row duplicates an earlier row exactly.")

    # Duplicate order IDs: flag all rows in a repeated non-empty ID group.
    valid_order = work["order_id"].map(lambda v: not is_missing(v))
    repeated_ids = work.loc[valid_order, "order_id"].astype(str).duplicated(keep=False)
    for idx in work.index[valid_order][repeated_ids]:
        add_issue(issues, idx, int(idx) + 2, "order_id", "duplicate_order_id",
                  work.at[idx, "order_id"], "This order_id appears on more than one row.")

    # Date format and validity: strict ISO YYYY-MM-DD.
    for idx, value in work["order_date"].items():
        if is_missing(value):
            continue
        text_value = str(value).strip()
        valid = bool(DATE_PATTERN.fullmatch(text_value))
        if valid:
            try:
                pd.to_datetime(text_value, format="%Y-%m-%d", errors="raise")
            except (ValueError, TypeError):
                valid = False
        if not valid:
            add_issue(issues, idx, int(idx) + 2, "order_date", "invalid_date_format",
                      value, "Expected a valid date in YYYY-MM-DD format.")

    # Category inconsistency. Flag non-approved raw categories; known aliases can be normalized.
    for idx, value in work["product_category"].items():
        if is_missing(value):
            continue
        raw = str(value)
        normalized_key = raw.strip().lower().replace(" ", "")
        alias_key = raw.strip().lower()
        canonical = CATEGORY_ALIASES.get(alias_key) or CATEGORY_ALIASES.get(normalized_key)
        if raw != raw.strip() or raw not in APPROVED_CATEGORIES:
            if canonical in APPROVED_CATEGORIES:
                message = f"Non-canonical category spelling/case; suggested value: {canonical}."
            else:
                message = "Category is not in the approved category list; review or extend the list."
            add_issue(issues, idx, int(idx) + 2, "product_category",
                      "inconsistent_category", value, message)

    # Numeric conversion and value checks.
    numeric_cols = ["quantity", "unit_price"]
    numeric_values: dict[str, pd.Series] = {}
    for col in numeric_cols:
        converted = pd.to_numeric(work[col].map(lambda v: np.nan if is_missing(v) else v),
                                  errors="coerce")
        numeric_values[col] = converted
        for idx, original in work[col].items():
            if is_missing(original):
                continue
            try:
                float(original)
            except (ValueError, TypeError):
                add_issue(issues, idx, int(idx) + 2, col, "invalid_numeric_format",
                          original, f"{col} must be numeric.")
                continue
            number = float(original)
            if col in {"quantity", "unit_price"} and number <= 0:
                add_issue(issues, idx, int(idx) + 2, col, "non_positive_numeric",
                          original, f"{col} should be greater than zero.")

    # IQR outlier rule, computed per column on numeric, finite, positive values.
    # This is a review flag, not proof that a value is incorrect.
    for col, series in numeric_values.items():
        candidates = series[np.isfinite(series) & (series > 0)].dropna()
        if len(candidates) < 4:
            continue
        q1, q3 = candidates.quantile([0.25, 0.75])
        iqr = q3 - q1
        # Use a conservative 3*IQR fence to reduce false alarms in naturally
        # skewed retail prices. This is still a review flag, not proof of an error.
        if iqr == 0:
            lower, upper = q1, q3
        else:
            lower, upper = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        for idx, value in candidates.items():
            if value < lower or value > upper:
                add_issue(issues, idx, int(idx) + 2, col, "iqr_outlier", value,
                          f"Value is outside the conservative 3*IQR review bounds [{lower:.3g}, {upper:.3g}].")

    # A conservative review copy: trim whitespace and canonicalize recognized categories.
    review = work.copy()
    for col in review.columns:
        if review[col].dtype == object:
            review[col] = review[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    def canonicalize(value: Any) -> Any:
        if is_missing(value):
            return value
        key = str(value).strip().lower()
        compact = key.replace(" ", "")
        return CATEGORY_ALIASES.get(key) or CATEGORY_ALIASES.get(compact) or value
    review["product_category"] = review["product_category"].map(canonicalize)

    issue_df = pd.DataFrame(issues)
    counts = issue_df["rule"].value_counts().to_dict() if not issue_df.empty else {}
    summary = {
        "rows_audited": int(len(work)),
        "columns_audited": [c for c in EXPECTED_COLUMNS],
        "total_findings": int(len(issues)),
        "rows_with_findings": int(issue_df["row_index"].nunique()) if not issue_df.empty else 0,
        "findings_by_rule": {str(k): int(v) for k, v in counts.items()},
        "approved_product_categories": sorted(APPROVED_CATEGORIES),
        "notes": [
            "Findings are review flags, not automatic proof of bad data.",
            "IQR outliers can be legitimate values.",
            "Repeated order IDs may be valid for some transaction schemas."
        ],
    }
    return review, issues, summary


def evaluate_issues(issues: list[dict[str, Any]], manifest_path: Path) -> dict[str, Any]:
    """Compare detected (row_index, column, rule) triples to known injected findings."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        (int(item["row_index"]), str(item["column"]), str(item["rule"]))
        for item in manifest["injected_errors"]
    }
    detected = {
        (int(item["row_index"]), str(item["column"]), str(item["rule"]))
        for item in issues
    }
    true_positive = len(expected & detected)
    false_negative = len(expected - detected)
    false_positive = len(detected - expected)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    return {
        "expected_injected_findings": len(expected),
        "detected_findings": len(detected),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round((2 * precision * recall / (precision + recall)), 4)
                    if precision + recall else 0.0,
        "interpretation": "Metrics are specific to this synthetic test set; review broad rules such as outlier detection for false positives."
    }


def run_audit(input_path: Path, out_dir: Path, evaluate_path: Path | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path, dtype=object, keep_default_na=False)
    review, issues, summary = audit_dataframe(df)
    summary["input_file"] = str(input_path)
    if evaluate_path:
        summary["evaluation"] = evaluate_issues(issues, evaluate_path)

    issue_columns = ["row_index", "csv_row_number", "column", "rule", "value", "message"]
    pd.DataFrame(issues, columns=issue_columns).to_csv(out_dir / "issues.csv", index=False)
    review.to_csv(out_dir / "cleaned_review_copy.csv", index=False)
    (out_dir / "quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a retail CSV for common data-quality issues.")
    parser.add_argument("--input", required=True, type=Path, help="Path to input CSV.")
    parser.add_argument("--out", default=Path("reports"), type=Path, help="Output directory.")
    parser.add_argument("--evaluate", type=Path, help="Optional JSON manifest of known injected errors.")
    args = parser.parse_args()
    summary = run_audit(args.input, args.out, args.evaluate)
    print(json.dumps(summary, indent=2))
    print(f"\nReports written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
