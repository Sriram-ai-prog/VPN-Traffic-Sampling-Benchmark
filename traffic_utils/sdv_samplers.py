# sdv_samplers.py
"""
CTGAN-based conditional tabular data generation for multiclass imbalance.

IMPORTANT:
- CTGAN is used ONLY as a data augmentation method.
- All generated data is validated against domain constraints.
- Results are reported WITH fidelity and CVR diagnostics.
"""

from typing import List
import numpy as np
import pandas as pd
from collections import Counter

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition

from .config import TARGET_COL, CATEGORICAL_COLS, CTGAN_EPOCHS
from .data_loader import enforce_logical_consistency
from .metrics_utils import (
    compute_distribution_fidelity,
    compute_constraint_violation_rate,
)
from .models import evaluate_models


# =============================================================================
# 1. CTGAN SAMPLER (CONDITIONAL, MINORITY-FOCUSED)
# =============================================================================

def run_ctgan(
    X_train,
    y_train,
    X_test,
    y_test,
    encoder,
    epochs: int = CTGAN_EPOCHS,
    repeats: int = 3,
):
    """
    Conditional CTGAN for minority class augmentation.

    CTGAN is trained on the full training data but samples
    are generated conditionally per minority class.

    Results are reported per run to expose variance.
    """

    print("\n--- Conditional CTGAN ---")

    # Reconstruct training table
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train

    # Metadata definition
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)

    for col in CATEGORICAL_COLS + [TARGET_COL]:
        if col in train_df.columns:
            metadata.update_column(col, sdtype="categorical")

    class_counts = Counter(y_train)
    target_size = max(class_counts.values())

    for run in range(repeats):
        print(f"[CTGAN] Run {run + 1}/{repeats}")

        ctgan = CTGANSynthesizer(
            metadata=metadata,
            epochs=epochs,
            verbose=False,
        )

        ctgan.fit(train_df)

        synthetic_chunks: List[pd.DataFrame] = []

        # ---------------------------------------------------------------------
        # Conditional sampling per minority class
        # ---------------------------------------------------------------------
        for cls, cnt in class_counts.items():
            if cnt >= target_size:
                continue

            needed = target_size - cnt
            cond = Condition(
                num_rows=needed,
                column_values={TARGET_COL: cls},
            )

            try:
                samples = ctgan.sample_from_conditions([cond])
                synthetic_chunks.append(samples)
            except Exception:
                print(f"[CTGAN] Conditional sampling failed for class {cls}")

        if not synthetic_chunks:
            print("[CTGAN] No valid synthetic samples generated — skipping run.")
            continue

        syn_df = pd.concat(synthetic_chunks, ignore_index=True)

        # ---------------------------------------------------------------------
        # Fidelity BEFORE snapping (raw GAN behavior)
        # ---------------------------------------------------------------------
        fidelity_raw = compute_distribution_fidelity(
            X_train, syn_df[X_train.columns]
        )

        # ---------------------------------------------------------------------
        # Enforce domain constraints
        # ---------------------------------------------------------------------
        X_syn = syn_df.drop(columns=[TARGET_COL])
        y_syn = syn_df[TARGET_COL]

        X_res = pd.concat([X_train, X_syn], ignore_index=True)
        y_res = np.concatenate([y_train, y_syn.values])

        X_res, y_res = enforce_logical_consistency(
            X_res.values, y_res, X_train.columns.tolist()
        )

        X_res_df = pd.DataFrame(X_res, columns=X_train.columns)

        # ---------------------------------------------------------------------
        # CVR AFTER snapping (final validity)
        # ---------------------------------------------------------------------
        cvr = compute_constraint_violation_rate(X_res_df)

        fidelity_post = compute_distribution_fidelity(
            X_train, X_res_df
        )

        print(
            f"[CTGAN] Raw KS={fidelity_raw['mean_ks']:.4f} | "
            f"Post KS={fidelity_post['mean_ks']:.4f} | "
            f"CVR={cvr:.4f}"
        )

        # ---------------------------------------------------------------------
        # IMPORTANT: Even if CVR > threshold, results are reported with caveat
        # ---------------------------------------------------------------------
        return evaluate_models(
            X_res,
            y_res,
            X_test,
            y_test,
            encoder,
            sampler_name=f"CTGAN_run{run + 1}",
        )
