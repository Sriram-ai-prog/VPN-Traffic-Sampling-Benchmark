# visualizations.py
"""
Visualization utilities matched to user requirements.
1. Decision Boundary: Pastel regions, black-outlined points, custom legend.
2. Probability Heatmap: Grid layout, blue-to-red landscapes, per-class focus.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from sklearn.base import clone
from typing import Optional, List


def plot_confusion_matrix(model, X_test, y_test, class_names=None, title="Confusion Matrix", cmap="viridis", ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=class_names, cmap=cmap,
        xticks_rotation="vertical", ax=ax, colorbar=True
    )
    disp.ax_.set_title(title)
    return disp


def plot_decision_boundary_2d(model, X, y, class_names=None, feature_indices=None, use_pca=False,
                              title="Decision Boundary (2D)", ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(12, 8))

    # 1. Project to 2D
    if X.shape[1] > 2:
        if use_pca:
            pca = PCA(n_components=2)
            X_plot = pca.fit_transform(X)
            xlabel, ylabel = "Principal Component 1", "Principal Component 2"
        elif feature_indices:
            X_plot = X[:, feature_indices]
            xlabel, ylabel = f"Feat {feature_indices[0]}", f"Feat {feature_indices[1]}"
        else:
            raise ValueError("Use PCA or feature indices for >2D data.")

        viz_model = clone(model)
        if hasattr(viz_model, "n_jobs"): viz_model.n_jobs = 1
        viz_model.fit(X_plot, y)
    else:
        X_plot, viz_model = X, model
        xlabel, ylabel = "Feature 0", "Feature 1"

    # 2. Pastel Regions (tab20)
    DecisionBoundaryDisplay.from_estimator(
        viz_model, X_plot, response_method="predict", plot_method="pcolormesh",
        shading="auto", alpha=0.3, cmap="tab20", ax=ax
    )

    # 3. Scatter with Legend
    unique_y = np.unique(y)
    cmap = plt.cm.get_cmap("tab20", len(unique_y))

    for i, cls in enumerate(unique_y):
        mask = (y == cls)
        lbl = f"Class: {class_names[cls]}" if class_names is not None else f"Class: {cls}"
        ax.scatter(X_plot[mask, 0], X_plot[mask, 1], label=lbl, color=cmap(i), edgecolor="k", s=40, alpha=0.9)

    ax.set_title(title);
    ax.set_xlabel(xlabel);
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    return ax


def plot_probability_heatmap(model, X, y, class_names=None, feature_indices=None, use_pca=False,
                             title="Probability Landscapes"):
    """
    Grid-based Probability Landscapes matching the reference image.
    Blue (0.0) -> Red (1.0) probability contours.
    """
    # 1. Prepare 2D Data
    if X.shape[1] > 2:
        if use_pca:
            pca = PCA(n_components=2)
            X_plot = pca.fit_transform(X)
        elif feature_indices:
            X_plot = X[:, feature_indices]
        else:
            raise ValueError("Requires PCA or feature indices.")

        viz_model = clone(model)
        if hasattr(viz_model, "n_jobs"): viz_model.n_jobs = 1
        viz_model.fit(X_plot, y)
    else:
        X_plot, viz_model = X, model

    # 2. Setup Grid
    classes = np.unique(y)
    n_classes = len(classes)
    cols = 3
    rows = int(np.ceil(n_classes / cols))

    fig = plt.figure(figsize=(15, 4 * rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig)

    # 3. Plot per class
    for idx, cls in enumerate(classes):
        ax = fig.add_subplot(gs[idx])
        cls_name = class_names[cls] if class_names is not None else str(cls)

        # A. Probability Contour (Blue -> Red)
        try:
            DecisionBoundaryDisplay.from_estimator(
                viz_model, X_plot, response_method="predict_proba", class_of_interest=cls,
                plot_method="contourf", cmap="seismic", alpha=0.8, ax=ax
            )
        except:
            # Fallback for old sklearn
            print(f"Skipping prob plot for {cls} (sklearn version issue)")

        # B. Scatter Points (Target vs Others)
        others_mask = (y != cls)
        target_mask = (y == cls)

        # Plot Others (Black, faded)
        ax.scatter(X_plot[others_mask, 0], X_plot[others_mask, 1], c="black", s=10, alpha=0.2, label="Other Classes")

        # Plot Target (Bright Orange/Gold with white edge, similar to image)
        ax.scatter(X_plot[target_mask, 0], X_plot[target_mask, 1], c="gold", edgecolor="white", s=40,
                   label=f"Actual: {cls_name}")

        ax.set_title(f"Prob for: {cls_name}")
        ax.legend(fontsize="x-small", loc="upper right")
        ax.set_xticks([]);
        ax.set_yticks([])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig