"""Automated pipeline to fetch official open data from OpenDOSM / data.gov.my

and synthesize Malaysian retail transactions for benchmarking.
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
import requests

MALAYSIAN_STATES = [
    "Selangor",
    "Johor",
    "Penang",
    "W.P. Kuala Lumpur",
    "Perak",
    "Kedah",
    "Perlis",
    "Melaka",
    "Negeri Sembilan",
    "Pahang",
    "Terengganu",
    "Kelantan",
    "Sarawak",
    "Sabah",
]

PAYMENT_METHODS = ["DuitNow QR", "Touch 'n Go eWallet", "FPX", "Credit Card"]
CATEGORIES = [
    "Groceries & Food",
    "Consumer Electronics",
    "Fashion & Apparel",
    "Health & Beauty",
    "Home & Living",
]


def fetch_opendosm_macro_retail(output_dir: Path):
    """Downloads real monthly wholesale & retail trade indices from OpenDOSM."""
    url = "https://storage.dosm.gov.my/iowrt/iowrt.csv"
    dest_path = output_dir / "opendosm_wholesale_retail.csv"

    print(f"📡 Fetching official DOSM data from {url}...")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Saved official DOSM dataset to: {dest_path}")
    except Exception as e:
        print(f"⚠️ Could not download live OpenDOSM dataset: {e}")


def generate_malaysia_retail_transactions(
    output_dir: Path, n_records: int = 500
):
    """Generates realistic Malaysian retail transaction logs (MYR) with real

    state distributions and controlled data quality anomalies.
    """
    random.seed(42)
    np.random.seed(42)

    order_ids = [f"MY-{20260000 + i}" for i in range(n_records)]
    dates = pd.date_range("2026-01-01", periods=n_records, freq="h").strftime(
        "%Y-%m-%d %H:%M"
    )

    df = pd.DataFrame(
        {
            "order_id": order_ids,
            "transaction_date": dates,
            "customer_state": [
                random.choice(MALAYSIAN_STATES) for _ in range(n_records)
            ],
            "product_category": [
                random.choice(CATEGORIES) for _ in range(n_records)
            ],
            "quantity": [random.randint(1, 8) for _ in range(n_records)],
            "unit_price_myr": [
                round(random.uniform(10.0, 450.0), 2) for _ in range(n_records)
            ],
            "payment_method": [
                random.choice(PAYMENT_METHODS) for _ in range(n_records)
            ],
        }
    )

    # Injected defects for benchmarking
    # 1. Missing states/payment
    for idx in random.sample(range(10, 60), 8):
        df.at[idx, "customer_state"] = np.nan

    # 2. Corrupted payment method
    for idx in random.sample(range(70, 120), 5):
        df.at[idx, "payment_method"] = "Crypto/Unsupported"

    # 3. Duplicate Order IDs
    for idx in [150, 151, 152]:
        df.iloc[idx] = df.iloc[idx - 50]

    # 4. Price anomaly in MYR
    df.at[200, "unit_price_myr"] = 99999.00
    df.at[205, "quantity"] = -3

    output_path = output_dir / "malaysia_retail_transactions.csv"
    df.to_csv(output_path, index=False)
    print(
        f"✅ Generated Malaysian transaction audit dataset: {output_path} ({len(df)} rows)"
    )


if __name__ == "__main__":
    data_dir = (
        Path(__file__).resolve().parent.parent / "data"
    )  # D:\data_quality_guardian\data_quality_guardian\data
    data_dir.mkdir(exist_ok=True)
    fetch_opendosm_macro_retail(data_dir)
    generate_malaysia_retail_transactions(data_dir, n_records=600)
