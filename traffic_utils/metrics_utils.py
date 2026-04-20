# metrics_utils.py
"""
Metric utilities for the VPN traffic sampling benchmark.

This module provides:
- Classification performance metrics (Macro-F1 primary)
- Data fidelity metrics for synthetic samples
- Domain validity metrics (Constraint Violation Rate - CVR)
- Statistical testing utilities for research claims
"""

from typing import Dict, List
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, wilcoxon
from sklearn.metrics import classification_report, f1_score


# =============================================================================
# 1. HELPER: ENSURE DATAFRAME CONSISTENCY
# =============================================================================

def _ensure_dataframe(X, reference: pd.DataFrame = None) -> pd.DataFrame:
    """
    Convert input to DataFrame while preserving column names if possible.
    """
    if isinstance(X, pd.DataFrame):
        return X
    if reference is not None and isinstance(reference, pd.DataFrame):
        return pd.DataFrame(X, columns=reference.columns)
    if isinstance(X, np.ndarray):
        return pd.DataFrame(X, columns=([f"f_{i}" for i in range(X.shape[1])]))
    return pd.DataFrame(X)


# =============================================================================
# 2. DATA FIDELITY METRICS (SYNTHETIC vs REAL)
# =============================================================================

def compute_distribution_fidelity(
    real_X,
    syn_X,
    max_features: int = 50,
) -> Dict[str, float]:
    """
    Quantify how closely synthetic data matches real data distributions.

    Metrics:
    - mean KS statistic across numeric features
    - fraction of features with KS p-value > 0.05
    - mean absolute Pearson correlation difference

    These metrics are NOT performance metrics.
    They are diagnostic tools for data plausibility.
    """

    real = _ensure_dataframe(real_X)
    syn = _ensure_dataframe(syn_X, reference=real)

    common_cols = [c for c in real.columns if c in syn.columns]
    real = real[common_cols]
    syn = syn[common_cols]

    numeric_cols = real.select_dtypes(include=[np.number]).columns.tolist()
    if max_features and len(numeric_cols) > max_features:
        numeric_cols = numeric_cols[:max_features]

    ks_stats, ks_pvals = [], []

    for col in numeric_cols:
        r = real[col].dropna()
        s = syn[col].dropna()
        if len(r) > 20 and len(s) > 20:
            stat, pval = ks_2samp(r, s)
            ks_stats.append(stat)
            ks_pvals.append(pval)

    mean_ks = float(np.mean(ks_stats)) if ks_stats else np.nan
    frac_p_gt_005 = float(np.mean(np.array(ks_pvals) > 0.05)) if ks_pvals else np.nan

    if len(numeric_cols) >= 2:
        corr_real = real[numeric_cols].corr().values
        corr_syn = syn[numeric_cols].corr().values
        mean_corr_diff = float(np.mean(np.abs(corr_real - corr_syn)))
    else:
        mean_corr_diff = np.nan

    return {
        "mean_ks": mean_ks,
        "frac_features_p_gt_0_05": frac_p_gt_005,
        "mean_corr_diff": mean_corr_diff,
    }


# =============================================================================
# 3. DOMAIN VALIDITY METRIC (CRITICAL)
# =============================================================================

def compute_constraint_violation_rate(
    X: pd.DataFrame,
) -> float:
    """
    Compute Constraint Violation Rate (CVR).

    CVR = fraction of samples violating domain rules.

    This metric is mandatory for:
    - GAN-based samplers
    - Structural augmentation methods
    """

    violations = np.zeros(len(X), dtype=bool)

    # Rule 1: Non-negativity of time-based features
    nonneg_cols = [
        "duration",
        "min_fiat", "max_fiat", "mean_fiat",
        "min_biat", "max_biat", "mean_biat",
        "min_flowiat", "max_flowiat", "mean_flowiat",
        "min_active", "max_active", "mean_active",
        "min_idle", "max_idle", "mean_idle",
    ]
    for col in nonneg_cols:
        if col in X.columns:
            violations |= X[col] < 0

    # Rule 2: Continuous flow ⇒ idle time must be zero
    if "is_continuous_flow" in X.columns:
        mask = X["is_continuous_flow"] == 1
        idle_cols = [c for c in ["min_idle", "max_idle", "mean_idle"] if c in X.columns]
        for col in idle_cols:
            violations |= mask & (X[col] > 1e-6)

    # Rule 3: min ≤ mean ≤ max consistency
    def _check_order(min_col, mean_col, max_col):
        if min_col in X.columns and mean_col in X.columns and max_col in X.columns:
            return ~(
                (X[min_col] <= X[mean_col]) &
                (X[mean_col] <= X[max_col])
            )
        return np.zeros(len(X), dtype=bool)

    violations |= _check_order("min_active", "mean_active", "max_active")
    violations |= _check_order("min_idle", "mean_idle", "max_idle")
    violations |= _check_order("min_fiat", "mean_fiat", "max_fiat")
    violations |= _check_order("min_biat", "mean_biat", "max_biat")

    return float(np.mean(violations))


# =============================================================================
# 4. CLASSIFICATION PERFORMANCE METRICS
# =============================================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder,
) -> Dict:
    """
    Compute classification metrics for reporting.

    Primary metric:
    - Macro-F1 (treats all classes equally)

    Secondary:
    - Weighted-F1
    - Full per-class precision / recall / F1
    """

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    per_class = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_report": per_class,
    }


# =============================================================================
# 5. STATISTICAL SIGNIFICANCE TESTING
# =============================================================================

def wilcoxon_signed_rank_test(
    scores_a: List[float],
    scores_b: List[float],
) -> Dict[str, float]:
    """
    Paired Wilcoxon signed-rank test.

    Used to compare two sampling techniques across
    identical seeds/splits.

    Returns:
    - statistic
    - p_value
    """

    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have the same length.")

    stat, p_value = wilcoxon(scores_a, scores_b)

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
    }
