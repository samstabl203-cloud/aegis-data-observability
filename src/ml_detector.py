"""Machine Learning Anomaly Detection Engine using Isolation Forest.

Detects multivariate anomalies across numerical features and category contexts.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


class MLAnomalyDetector:

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        self.scaler = RobustScaler()

    def fit_predict(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Runs multivariate anomaly detection on numerical attributes.

        Features examined: quantity, unit_price, total_amount (interaction
        term).
        """
        data = df.copy()

        # Parse numerical columns cleanly
        data["clean_qty"] = pd.to_numeric(data["quantity"], errors="coerce")
        data["clean_price"] = pd.to_numeric(data["unit_price"], errors="coerce")

        # Impute temporary median values for unparseable entries so ML model can run
        data["clean_qty"] = data["clean_qty"].fillna(data["clean_qty"].median())
        data["clean_price"] = data["clean_price"].fillna(
            data["clean_price"].median()
        )

        # Feature engineering: interaction feature
        data["total_transaction_val"] = data["clean_qty"] * data["clean_price"]

        feature_cols = ["clean_qty", "clean_price", "total_transaction_val"]
        features = data[feature_cols]

        scaled_features = self.scaler.fit_transform(features)
        # Isolation Forest outputs -1 for anomaly, 1 for normal
        predictions = self.model.fit_predict(scaled_features)
        anomaly_scores = self.model.decision_function(scaled_features)

        is_anomaly = predictions == -1
        results_df = pd.DataFrame(
            {
                "row_index": df.index,
                "is_ml_anomaly": is_anomaly,
                "anomaly_score": np.round(anomaly_scores, 4),
            }
        )

        return is_anomaly, results_df
