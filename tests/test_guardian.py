import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.guardian import audit_dataframe, evaluate_issues


class TestDataQualityGuardian(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            {"order_id": "O100", "order_date": "2026-01-01", "customer_id": "C1",
             "product_category": "Electronics", "quantity": "2", "unit_price": "10.00", "region": "North"},
            {"order_id": "O101", "order_date": "2026-01-02", "customer_id": "C2",
             "product_category": "Home Decor", "quantity": "1", "unit_price": "20.00", "region": "South"},
            {"order_id": "O102", "order_date": "2026-01-03", "customer_id": "C3",
             "product_category": "Clothing", "quantity": "3", "unit_price": "15.00", "region": "East"},
            {"order_id": "O103", "order_date": "2026-01-04", "customer_id": "C4",
             "product_category": "Groceries", "quantity": "4", "unit_price": "12.00", "region": "West"},
        ])

    def test_clean_data_has_no_findings(self):
        _, issues, _ = audit_dataframe(self.df)
        # With only four ordinary values per numeric column, no outlier should be detected here.
        self.assertEqual(issues, [])

    def test_detects_missing_date_category_and_nonpositive_quantity(self):
        df = self.df.copy()
        df.loc[0, "customer_id"] = ""
        df.loc[1, "order_date"] = "02/01/2026"
        df.loc[2, "product_category"] = " home decor "
        df.loc[3, "quantity"] = "-1"
        _, issues, _ = audit_dataframe(df)
        rules = {(x["row_index"], x["rule"]) for x in issues}
        self.assertIn((0, "missing_value"), rules)
        self.assertIn((1, "invalid_date_format"), rules)
        self.assertIn((2, "inconsistent_category"), rules)
        self.assertIn((3, "non_positive_numeric"), rules)

    def test_detects_duplicate_row_and_duplicate_order_id(self):
        df = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        _, issues, _ = audit_dataframe(df)
        rules = {x["rule"] for x in issues}
        self.assertIn("duplicate_row", rules)
        self.assertIn("duplicate_order_id", rules)

    def test_evaluation_metrics(self):
        issues = [
            {"row_index": 0, "column": "customer_id", "rule": "missing_value"},
            {"row_index": 1, "column": "order_date", "rule": "invalid_date_format"},
        ]
        manifest = {
            "injected_errors": [
                {"row_index": 0, "column": "customer_id", "rule": "missing_value"},
                {"row_index": 1, "column": "order_date", "rule": "invalid_date_format"},
                {"row_index": 2, "column": "quantity", "rule": "non_positive_numeric"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            result = evaluate_issues(issues, path)
        self.assertEqual(result["true_positives"], 2)
        self.assertEqual(result["false_negatives"], 1)
        self.assertEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 0.6667, places=4)


if __name__ == "__main__":
    unittest.main()
