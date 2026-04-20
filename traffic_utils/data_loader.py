# data_loader.py
"""
Data loading and splitting utilities for the VPN traffic sampling benchmark.

Design principles:
- Absolutely no data leakage
- Splitting occurs BEFORE any statistical operation
- Compatible with sampling-inside-CV experiments
- Explicit and auditable behavior for reviewers
"""

from typing import Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from .config import TARGET_COL, CATEGORICAL_COLS, SEED


# =============================================================================
# 1. RAW DATA LOADING
# =============================================================================

def load_raw_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the raw dataset without performing ANY statistical fitting.

    This function:
    - Reads the CSV
    - Drops non-informative index columns
    - Removes rows with missing labels ONLY
    - Performs NO imputation, scaling, or encoding

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (raw, untouched)
    y : pd.Series
        Target labels (raw categorical)
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dataset not found: {filepath}") from e

    # Drop accidental index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Drop rows without target label
    df = df.dropna(subset=[TARGET_COL])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(str)

    return X, y


# =============================================================================
# 2. TRAIN / VALIDATION / TEST SPLITTING (LEAKAGE-SAFE)
# =============================================================================

def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> Tuple:
    """
    Stratified Train / Validation / Test split.

    IMPORTANT:
    - Splitting is performed BEFORE any imputation or encoding
    - Validation and test sets remain strictly untouched

    Returns
    -------
    X_train, X_val, X_test : pd.DataFrame
    y_train, y_val, y_test : pd.Series
    label_encoder : LabelEncoder (fitted ONLY on y_train)
    """

    # --- First split: Train+Val vs Test ---
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=SEED,
    )

    # --- Second split: Train vs Val ---
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val_size,
        stratify=y_temp,
        random_state=SEED,
    )

    # Encode labels (fit ONLY on training labels)
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train_enc,
        y_val_enc,
        y_test_enc,
        label_encoder,
    )


# =============================================================================
# 3. NUMERIC IMPUTATION (TRAIN-ONLY FIT)
# =============================================================================

def impute_numeric_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Median imputation fitted ONLY on training data.

    This function exists separately so that:
    - Imputation can be performed inside CV folds if needed
    - Reviewers can verify no leakage occurred
    """

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)

    X_train_imp = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Restore categorical dtypes explicitly
    for col in CATEGORICAL_COLS:
        if col in X_train_imp.columns:
            X_train_imp[col] = X_train_imp[col].round().astype(int)
            X_val_imp[col] = X_val_imp[col].round().astype(int)
            X_test_imp[col] = X_test_imp[col].round().astype(int)

    return X_train_imp, X_val_imp, X_test_imp


# =============================================================================
# 4. DOMAIN LOGICAL CONSISTENCY ENFORCEMENT
# =============================================================================

def enforce_logical_consistency(X_resampled, y_resampled, feature_names):
    df = pd.DataFrame(X_resampled, columns=feature_names)

    # 1. CLIP NEGATIVES (First, get rid of impossible time)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].clip(lower=0.0)

    # 2. ROUND INTEGERS (Fix the "1.5 packets" issue)
    # Note: We round BEFORE fixing Min/Max logic, so we catch the "Rounding Trap"
    integer_cols = [
        "duration", "total_fiat", "total_biat",
        "min_fiat", "max_fiat", "mean_fiat",  # Round mean if strictly integer, otherwise keep float
        "min_biat", "max_biat", "mean_biat",
        "min_flowiat", "max_flowiat", "mean_flowiat",
        "min_active", "max_active", "mean_active",
        "min_idle", "max_idle", "mean_idle",
        "flowPktsPerSecond", "flowBytesPerSecond"  # Usually float, check your data
    ]
    # Filter to only cols that actually exist
    existing_int_cols = [c for c in integer_cols if c in df.columns]
    df[existing_int_cols] = df[existing_int_cols].round()

    # 3. FIX MIN/MAX LOGIC (The "Swap" Fix)
    # Now that numbers are rounded, we ensure Min <= Max
    min_max_pairs = [
        ("min_fiat", "max_fiat"),
        ("min_biat", "max_biat"),
        ("min_flowiat", "max_flowiat"),
        ("min_active", "max_active"),
        ("min_idle", "max_idle"),
    ]
    for min_c, max_c in min_max_pairs:
        if min_c in df.columns and max_c in df.columns:
            # If min > max, swap them
            mask = df[min_c] > df[max_c]
            if mask.any():
                temp = df.loc[mask, min_c].copy()
                df.loc[mask, min_c] = df.loc[mask, max_c]
                df.loc[mask, max_c] = temp

    # 4. FIX MEAN LOGIC (Clip Mean between Min and Max)
    triplets = [
        ("min_fiat", "mean_fiat", "max_fiat"),
        ("min_biat", "mean_biat", "max_biat"),
        ("min_flowiat", "mean_flowiat", "max_flowiat"),
        ("min_active", "mean_active", "max_active"),
        ("min_idle", "mean_idle", "max_idle"),
    ]
    for min_c, mean_c, max_c in triplets:
        if min_c in df.columns and mean_c in df.columns and max_c in df.columns:
            df[mean_c] = df[mean_c].clip(lower=df[min_c], upper=df[max_c])

    # 5. FIX DOMAIN RULES (Continuous Flow)
    if "is_continuous_flow" in df.columns:
        # Round the flag first! SMOTE might output 0.6
        df["is_continuous_flow"] = df["is_continuous_flow"].round().clip(0, 1)

        # Now force the rule
        mask = df["is_continuous_flow"] == 1
        idle_cols = [c for c in ["min_idle", "max_idle", "mean_idle", "std_idle"] if c in df.columns]
        if idle_cols:
            df.loc[mask, idle_cols] = 0.0

    # 6. FIX PACKET COUNTS (Directional)
    # If no_forward_packets=1, forward stats must be 0
    if "no_forward_packets" in df.columns:
        df["no_forward_packets"] = df["no_forward_packets"].round().clip(0, 1)
        mask = df["no_forward_packets"] == 1
        fiat_cols = [c for c in ["min_fiat", "max_fiat", "mean_fiat", "total_fiat"] if c in df.columns]
        if fiat_cols:
            df.loc[mask, fiat_cols] = 0.0

    if "no_backward_packets" in df.columns:
        df["no_backward_packets"] = df["no_backward_packets"].round().clip(0, 1)
        mask = df["no_backward_packets"] == 1
        biat_cols = [c for c in ["min_biat", "max_biat", "mean_biat", "total_biat"] if c in df.columns]
        if biat_cols:
            df.loc[mask, biat_cols] = 0.0

    return df.values, y_resampled