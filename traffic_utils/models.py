# models.py
"""
Model training, hyperparameter optimization, and evaluation.

Updates:
- Added 'RobustXGBClassifier' to fix VotingClassifier 'str vs int' errors.
- Added Training vs Test Accuracy reporting to detect overfitting.
- Optimized for Macro-F1 score (best for multiclass imbalance).
"""

from typing import Dict, List, Tuple
import numpy as np
import optuna
import xgboost as xgb
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score  # <--- Added for overfitting check
)

from .config import SEED, OPTUNA_TRIALS, CV_FOLDS
from .metrics_utils import compute_classification_metrics

# Suppress Optuna logging to keep output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# 1. HELPER: CONFUSION FOCUS
# =============================================================================

def _print_confusion_focus(y_true, y_pred, classes, title="Confusion Focus"):
    """Identifies and prints the most common misclassification for each class."""

    # Ensure labels are indices if y is integer
    if np.issubdtype(y_true.dtype, np.integer) or np.issubdtype(y_pred.dtype, np.integer):
        labels_for_cm = np.arange(len(classes))
    else:
        labels_for_cm = classes

    cm = confusion_matrix(y_true, y_pred, labels=labels_for_cm)

    print(f"\n[{title}]")
    print(f"{'Class':<20} | {'Most Common Confusion':<30} | {'Count':<5}")
    print("-" * 65)

    for idx, cls_name in enumerate(classes):
        row = cm[idx].copy()
        row[idx] = -1  # Ignore correct predictions
        most_confused_idx = np.argmax(row)
        error_count = row[most_confused_idx]

        if error_count > 0:
            confused_with = classes[most_confused_idx]
            print(f"{cls_name:<20} | {confused_with:<30} | {error_count:<5}")
    print("-" * 65)


# =============================================================================
# 2. THE FIX: ROBUST XGB CLASSIFIER (Property Override)
# =============================================================================

class RobustXGBClassifier(xgb.XGBClassifier):
    """
    A custom wrapper around XGBoost to prevent TypeErrors in Ensemble models.

    Fixes:
    1. 'str' vs 'int' comparison error in VotingClassifier.
    2. 'AttributeError: has no setter' by overriding the property getter.
    """

    def fit(self, X, y, **kwargs):
        # 1. Train normally
        super().fit(X, y, **kwargs)

        # 2. Save classes as integers in a private attribute
        # We rely on the parent's classes_ but cast it safely here.
        if hasattr(super(), "classes_"):
            self._robust_classes = super().classes_.astype(int)
        else:
            # Fallback: calculate from y directly if super doesn't expose it
            self._robust_classes = np.unique(y).astype(int)

        return self

    @property
    def classes_(self):
        """
        Override the parent property.
        If we have our robust integer classes, return them.
        Otherwise, ask the parent.
        """
        if hasattr(self, "_robust_classes"):
            return self._robust_classes
        return super().classes_


# =============================================================================
# 3. HYPERPARAMETER OPTIMIZATION
# =============================================================================

def tune_model_optuna(model_name: str, X: np.ndarray, y: np.ndarray) -> Dict:
    def objective(trial):
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

        if model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 6, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
                "max_features": "sqrt",
                "n_jobs": -1,
                "random_state": SEED
            }
            model = RandomForestClassifier(**params)

        elif model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "eval_metric": "mlogloss",
                "n_jobs": -1,
                "random_state": SEED,
            }
            # Use the Robust wrapper for tuning as well
            model = RobustXGBClassifier(**params)

        elif model_name == "HistGradient":
            params = {
                "max_iter": trial.suggest_int("max_iter", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 15),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 10.0),
                "random_state": SEED
            }
            model = HistGradientBoostingClassifier(**params)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    print(f" > Tuning {model_name}...", end=" ")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    print(f"Done. Best CV F1: {study.best_value:.4f}")

    print(f"   [Params] {study.best_params}")
    return study.best_params


# =============================================================================
# 4. MODEL FACTORY
# =============================================================================

def build_models(best_params: Dict[str, Dict]):
    # 1. Base Learners
    # Force n_jobs=-1 for parallelism

    rf_params = best_params["RandomForest"].copy()
    rf_params["n_jobs"] = -1
    rf = RandomForestClassifier(**rf_params, random_state=SEED)

    xgb_params = best_params["XGBoost"].copy()
    xgb_params["n_jobs"] = -1
    # USE THE ROBUST CLASS
    xgb_clf = RobustXGBClassifier(**xgb_params, random_state=SEED)

    hgb_params = best_params["HistGradient"].copy()
    hgb = HistGradientBoostingClassifier(**hgb_params, random_state=SEED)

    # 2. Ensembles
    voting = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb_clf), ("hgb", hgb)],
        voting="soft",
        n_jobs=-1
    )

    stacking = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb_clf), ("hgb", hgb)],
        final_estimator=HistGradientBoostingClassifier(max_iter=100, random_state=SEED),
        n_jobs=-1
    )

    return {
        "RandomForest": rf,
        "XGBoost": xgb_clf,
        "HistGradient": hgb,
        "SoftVoting": voting,
        "Stacking": stacking
    }


# =============================================================================
# 5. FULL EVALUATION ROUTINE
# =============================================================================

def evaluate_models(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        encoder,
        sampler_name: str,
) -> Tuple[List[Dict], Dict]:
    print("\n" + "=" * 80)
    print(f"REPORT: [{sampler_name}] | Train Samples: {X_train.shape[0]}")
    print("=" * 80)

    # --- Tuning ---
    print("\n--- 2. MODEL TUNING (Reproducibility) ---")
    best_params = {
        "RandomForest": tune_model_optuna("RandomForest", X_train, y_train),
        "XGBoost": tune_model_optuna("XGBoost", X_train, y_train),
        "HistGradient": tune_model_optuna("HistGradient", X_train, y_train),
    }

    models = build_models(best_params)
    results = []

    print("\n--- 3. CLASSIFICATION RESULTS (Performance & Overfitting Check) ---")

    for name, model in models.items():
        print(f"\n>> Training {name}...")
        try:
            model.fit(X_train, y_train)

            # --- OVERFITTING CHECK START ---
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            # -------------------------------

            metrics = compute_classification_metrics(y_test, y_test_pred, encoder)
            macro_f1 = metrics["macro_f1"]
            weighted_f1 = metrics["weighted_f1"]

            print(f"[{sampler_name}] {name} RESULTS:")
            print(f"   Train Acc:   {train_acc:.4f}  (Use to check overfitting)")
            print(f"   Test Acc:    {test_acc:.4f}")
            print(f"   Macro-F1:    {macro_f1:.4f}")
            print(f"   Weighted-F1: {weighted_f1:.4f}")


            print("\n   [Detailed Report]")
            print(classification_report(
                 y_test, y_test_pred,
                 target_names=encoder.classes_,
                 digits=4,
                 zero_division=0
            ))

            _print_confusion_focus(
                y_test, y_test_pred,
                classes=encoder.classes_,
                title=f"Confusion Focus ({name})"
            )

            results.append({
                "sampler": sampler_name,
                "model": name,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "params": best_params.get(name, {})
            })

        except Exception as e:
            print(f"[ERROR] Failed to train/evaluate {name}: {e}")

    return results, models