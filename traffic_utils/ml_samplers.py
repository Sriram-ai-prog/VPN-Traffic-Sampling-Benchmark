# ml_samplers.py
"""
Classical ML-based sampling techniques for multiclass imbalance.

Design principles:
- No data leakage
- Sampling applied ONLY to training data
- Explicit enforcement of domain constraints
- Fidelity and validity metrics logged per sampler
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from collections import Counter

from imblearn.over_sampling import SMOTE, RandomOverSampler, KMeansSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans

from .config import SEED
from .data_loader import enforce_logical_consistency
from .metrics_utils import (
    compute_distribution_fidelity,
    compute_constraint_violation_rate,
)
from .models import evaluate_models


# =============================================================================
# 1. GENERIC SAMPLER WRAPPER
# =============================================================================

def _apply_sampler(
    sampler,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a sampler safely and enforce domain constraints.
    """

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    X_res, y_res = enforce_logical_consistency(
        X_res, y_res, X_train.columns.tolist()
    )
    return X_res, y_res


def _log_sampler_metrics(
    name: str,
    X_real: pd.DataFrame,
    X_res: np.ndarray,
) -> Dict[str, float]:
    """
    Compute fidelity and validity metrics for a sampler.
    """

    X_res_df = pd.DataFrame(X_res, columns=X_real.columns)

    fidelity = compute_distribution_fidelity(
        real_X=X_real,
        syn_X=X_res_df,
    )
    cvr = compute_constraint_violation_rate(X_res_df)

    print(
        f"[{name}] "
        f"mean_KS={fidelity['mean_ks']:.4f} | "
        f"p>0.05={fidelity['frac_features_p_gt_0_05']:.3f} | "
        f"corr_diff={fidelity['mean_corr_diff']:.4f} | "
        f"CVR={cvr:.4f}"
    )

    return {
        **fidelity,
        "cvr": cvr,
    }


# =============================================================================
# 2. SAMPLER IMPLEMENTATIONS
# =============================================================================

def run_random_oversampler(
    X_train, y_train, X_test, y_test, encoder
):
    print("\n--- Random Oversampling ---")

    sampler = RandomOverSampler(random_state=SEED)
    X_res, y_res = _apply_sampler(sampler, X_train, y_train)

    _log_sampler_metrics("RandomOverSampler", X_train, X_res)
    return evaluate_models(X_res, y_res, X_test, y_test, encoder, "RandomOverSampler")


def run_random_undersampler(
    X_train, y_train, X_test, y_test, encoder
):
    print("\n--- Random Undersampling ---")

    sampler = RandomUnderSampler(random_state=SEED)
    X_res, y_res = _apply_sampler(sampler, X_train, y_train)

    _log_sampler_metrics("RandomUnderSampler", X_train, X_res)
    return evaluate_models(X_res, y_res, X_test, y_test, encoder, "RandomUnderSampler")


def run_smote(
    X_train, y_train, X_test, y_test, encoder
):
    print("\n--- SMOTE ---")

    sampler = SMOTE(random_state=SEED, k_neighbors=5)
    X_res, y_res = _apply_sampler(sampler, X_train, y_train)

    _log_sampler_metrics("SMOTE", X_train, X_res)
    return evaluate_models(X_res, y_res, X_test, y_test, encoder, "SMOTE")


def run_smoteenn(
    X_train, y_train, X_test, y_test, encoder
):
    print("\n--- SMOTE-ENN ---")

    sampler = SMOTEENN(random_state=SEED)
    X_res, y_res = _apply_sampler(sampler, X_train, y_train)

    _log_sampler_metrics("SMOTE-ENN", X_train, X_res)
    return evaluate_models(X_res, y_res, X_test, y_test, encoder, "SMOTE-ENN")


def run_smotetomek(
    X_train, y_train, X_test, y_test, encoder
):
    print("\n--- SMOTE-Tomek ---")

    sampler = SMOTETomek(random_state=SEED)
    X_res, y_res = _apply_sampler(sampler, X_train, y_train)

    _log_sampler_metrics("SMOTE-Tomek", X_train, X_res)
    return evaluate_models(X_res, y_res, X_test, y_test, encoder, "SMOTE-Tomek")


def run_cluster_smote(
        X_train, y_train, X_test, y_test, encoder
):
    print("\n--- Cluster-based SMOTE ---")

    class_counts = Counter(y_train)
    min_class_size = min(class_counts.values())

    # 1. Determine safe cluster count (cannot be > samples)
    # We ensure at least 2 clusters, but not more than (samples - 1)
    n_clusters = max(2, min(50, min_class_size - 1))

    # 2. CRITICAL FIX for "No clusters found"
    # cluster_balance_threshold: The threshold for accepting a cluster.
    # Default is 0.1. We lower it to 0.005 to force it to accept very noisy clusters.
    sampler = KMeansSMOTE(
        kmeans_estimator=KMeans(
            n_clusters=n_clusters,
            random_state=SEED,
            n_init=10
        ),
        cluster_balance_threshold=0.005,  # <--- THE FIX
        random_state=SEED,
    )

    try:
        # We try to apply it. If it STILL fails (e.g. class size < 2), we fall back.
        X_res, y_res = _apply_sampler(sampler, X_train, y_train)
        _log_sampler_metrics("ClusterSMOTE", X_train, X_res)
        return evaluate_models(X_res, y_res, X_test, y_test, encoder, "ClusterSMOTE")

    except Exception as e:
        print(f"[ClusterSMOTE] Failed with {n_clusters} clusters: {e}")
        print(" > Falling back to Standard SMOTE for this run...")
        # Fallback to standard SMOTE so the experiment doesn't crash
        return run_smote(X_train, y_train, X_test, y_test, encoder)
