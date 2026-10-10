"""Automated Remediation and Cleansing Engine for Data Quality Guardian."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class AutoDataCleaner:

    def __init__(self, canonical_categories: List[str] = None):
        self.canonical_categories = canonical_categories or [
            "Clothing",
            "Electronics",
            "Groceries",
            "Home Decor",
            "Office Supplies",
        ]

    def clean_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Cleans dirty data and logs all remediation actions taken."""
        cleaned = df.copy()
        log = {
            "duplicates_removed": 0,
            "categories_standardized": 0,
            "dates_coerced": 0,
            "missing_values_imputed": 0,
            "non_positive_numerics_fixed": 0,
        }

        # 1. Remove duplicate order_id records (keep first occurrence)
        if "order_id" in cleaned.columns:
            initial_count = len(cleaned)
            cleaned = cleaned.drop_duplicates(subset=["order_id"], keep="first")
            log["duplicates_removed"] = initial_count - len(cleaned)

        # 2. Normalize and map categories
        if "product_category" in cleaned.columns:
            lookup = {cat.lower(): cat for cat in self.canonical_categories}
            for idx, val in cleaned["product_category"].items():
                if pd.notna(val) and isinstance(val, str):
                    val_clean = val.strip().lower()
                    if (
                        val_clean in lookup
                        and val != lookup[val_clean]
                    ):
                        cleaned.at[idx, "product_category"] = lookup[val_clean]
                        log["categories_standardized"] += 1

        # 3. Handle Dates
        if "order_date" in cleaned.columns:
            parsed_dates = pd.to_datetime(cleaned["order_date"], errors="coerce")
            invalid_dates = parsed_dates.isna() & cleaned["order_date"].notna()
            log["dates_coerced"] = int(invalid_dates.sum())
            # Impute forward-fill or default
            cleaned["order_date"] = parsed_dates.dt.strftime("%Y-%m-%d").fillna(
                "2026-01-01"
            )

        # 4. Standardize Numeric Columns
        for num_col in ["quantity", "unit_price"]:
            if num_col in cleaned.columns:
                cleaned[num_col] = pd.to_numeric(
                    cleaned[num_col], errors="coerce"
                )
                missing_num = cleaned[num_col].isna().sum()
                log["missing_values_imputed"] += int(missing_num)

                # Median imputation for missing values
                med = cleaned[num_col].median()
                cleaned[num_col] = cleaned[num_col].fillna(med)

                # Fix non-positive values
                negatives = cleaned[num_col] <= 0
                log["non_positive_numerics_fixed"] += int(negatives.sum())
                cleaned.loc[negatives, num_col] = med

        return cleaned, log
