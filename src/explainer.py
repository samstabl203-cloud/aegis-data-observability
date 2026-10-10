"""Explainable AI (XAI) diagnostic engine for AegisData anomalies."""

from typing import Dict, List
import pandas as pd


class AnomalyExplainer:
    """Generates human-readable, diagnostic root-cause explanations."""

    @staticmethod
    def explain_record(row: pd.Series, df: pd.DataFrame) -> List[str]:
        reasons = []

        # 1. Price checks
        try:
            price = float(row.get("unit_price", row.get("unit_price_myr", 0)))
            if price <= 0:
                reasons.append(
                    f"Invalid non-positive pricing ({price} <= 0)."
                )
            else:
                median_price = df[
                    df.columns.intersection(["unit_price", "unit_price_myr"])[0]
                ].median()
                if price > (median_price * 10):
                    reasons.append(
                        f"Extreme price deviation: RM {price:,.2f} is significantly above the global median (RM {median_price:,.2f})."
                    )
        except (ValueError, TypeError):
            reasons.append("Unparseable non-numeric price entry.")

        # 2. Quantity checks
        try:
            qty = float(row.get("quantity", 1))
            if qty <= 0:
                reasons.append(
                    f"Non-positive transaction quantity ({qty} units)."
                )
            elif qty > 100:
                reasons.append(f"High-frequency bulk quantity ({qty} units).")
        except (ValueError, TypeError):
            reasons.append("Unparseable quantity entry.")

        # 3. Categorical checks
        cat = str(row.get("product_category", ""))
        valid_cats = {
            "clothing",
            "electronics",
            "groceries",
            "home decor",
            "office supplies",
            "groceries & food",
            "consumer electronics",
            "fashion & apparel",
            "health & beauty",
            "home & living",
        }
        if cat.strip().lower() not in valid_cats and cat.strip() != "":
            reasons.append(
                f"Taxonomy mismatch: Category '{cat}' does not match registered retail categories."
            )

        # 4. ML Score Context
        score = row.get("anomaly_score", None)
        if score is not None and score < 0:
            reasons.append(
                f"Multivariate outlier: Isolation Forest score ({score}) indicates joint distribution rarity."
            )

        if not reasons:
            reasons.append(
                "Record matches standard baseline distributions."
            )

        return reasons
