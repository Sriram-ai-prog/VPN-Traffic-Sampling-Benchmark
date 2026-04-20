# __init__.py
"""
VPN Traffic Sampling Benchmark
==============================
Reproducible framework for evaluating ML/DL sampling on network traffic.
"""

from .config import SEED, TARGET_COL, CATEGORICAL_COLS, DEVICE

from .data_loader import (
    load_raw_data,
    split_train_val_test,
    impute_numeric_features,
    enforce_logical_consistency,
)

from .metrics_utils import (
    compute_classification_metrics,
    compute_distribution_fidelity,
    compute_constraint_violation_rate,
    wilcoxon_signed_rank_test,
)

from .ml_samplers import (
    run_random_oversampler,
    run_random_undersampler,
    run_smote,
    run_smoteenn,
    run_smotetomek,
    run_cluster_smote,
)

from .dl_samplers import (
    run_feature_jittering,
    run_wgan,
)

from .sdv_samplers import run_ctgan

from .models import evaluate_models, tune_model_optuna

from .visualizations import (
    plot_confusion_matrix,
    plot_decision_boundary_2d,
    plot_probability_heatmap, # Now exported
)

__all__ = [
    "SEED", "TARGET_COL", "CATEGORICAL_COLS", "DEVICE",
    "load_raw_data", "split_train_val_test", "impute_numeric_features",
    "enforce_logical_consistency",
    "compute_classification_metrics", "compute_distribution_fidelity",
    "compute_constraint_violation_rate", "wilcoxon_signed_rank_test",
    "run_random_oversampler", "run_random_undersampler",
    "run_smote", "run_smoteenn", "run_smotetomek", "run_cluster_smote",
    "run_feature_jittering", "run_wgan", "run_ctgan",
    "evaluate_models", "tune_model_optuna",
    "plot_confusion_matrix", "plot_decision_boundary_2d", "plot_probability_heatmap"
]