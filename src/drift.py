"""AegisData Drift & Distribution Stability Engine.

Computes Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats


class DataDriftEngine:
  """Evaluates statistical covariate drift between baseline and incoming tabular distributions."""

  @staticmethod
  def calculate_psi(
      expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10
  ) -> float:
    """Calculates the Population Stability Index (PSI) between two continuous distributions.

    PSI < 0.10: Stable / No significant shift
    0.10 <= PSI < 0.20: Moderate shift / Monitoring recommended
    PSI >= 0.20: Significant drift / Model retraining required
    """
    expected_clean = expected[
        ~np.isnan(expected) & ~np.isinf(expected) & (expected > 0)
    ]
    actual_clean = actual[~np.isnan(actual) & ~np.isinf(actual) & (actual > 0)]

    if len(expected_clean) < 10 or len(actual_clean) < 10:
      return 0.0

    # Determine quantile bins based on reference/expected distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected_clean, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Calculate bucket frequencies
    expected_counts, _ = np.histogram(expected_clean, bins=bins)
    actual_counts, _ = np.histogram(actual_clean, bins=bins)

    # Convert to fractions with Laplace-style smoothing to prevent division by zero
    expected_pct = (expected_counts + 1e-4) / (len(expected_clean) + 1e-3)
    actual_pct = (actual_counts + 1e-4) / (len(actual_clean) + 1e-3)

    psi_val = np.sum(
        (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    )
    return float(np.round(max(0.0, psi_val), 4))

  @classmethod
  def evaluate_drift(
      cls,
      baseline_df: pd.DataFrame,
      target_df: pd.DataFrame,
      numeric_col: str,
  ) -> Dict[str, any]:
    """Runs dual-test evaluation (PSI and KS-Test) for a selected feature."""
    if (
        numeric_col not in baseline_df.columns
        or numeric_col not in target_df.columns
    ):
      return {
          "status": "Incompatible Columns",
          "psi": 0.0,
          "ks_statistic": 0.0,
          "p_value": 1.0,
          "verdict": "Column missing in one or both datasets",
      }

    base_vals = pd.to_numeric(baseline_df[numeric_col], errors="coerce").dropna(
    ).values
    target_vals = pd.to_numeric(target_df[numeric_col], errors="coerce").dropna(
    ).values

    psi_score = cls.calculate_psi(base_vals, target_vals)
    ks_result = stats.ks_2samp(base_vals, target_vals)

    if psi_score < 0.10:
      verdict = "🟢 Stable Distribution (No Drift Detected)"
      badge_color = "green"
    elif psi_score < 0.20:
      verdict = "🟡 Moderate Covariate Shift (Monitor Pipeline)"
      badge_color = "orange"
    else:
      verdict = "🔴 Significant Drift Alert (Retraining / Audit Mandatory)"
      badge_color = "red"

    return {
        "psi": psi_score,
        "ks_statistic": float(np.round(ks_result.statistic, 4)),
        "p_value": float(np.round(ks_result.pvalue, 4)),
        "verdict": verdict,
        "badge_color": badge_color,
    }
