# evaluation_helpers.py
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, Sequence, List
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def decision_boundary_plot(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,                  # raw 2D data (for limits + scatter)
    y: np.ndarray,                  # labels
    save_path: str,
    title: str = "Decision boundary",
    *,
    # Choose ONE of these two ways to set mesh resolution:
    n: Optional[int] = None,        # fixed mesh per axis (e.g., 150)
    budget: Optional[int] = None,   # cap ~ n^2 * sv_count (e.g., 500_000)
    sv_count: Optional[int] = None, # required if using budget (e.g., len(model.support_))
    # Aesthetics / limits
    pad: float = 0.15,
    min_n: int = 35,
    max_n: int = 300,
    scatter_alpha: float = 0.9,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> "matplotlib.figure.Figure":
    """
    Generic decision-boundary plotter for BOTH classical and quantum models.
    - You supply `predict_fn(grid)` that returns labels for a (M x 2) grid.
    - If `n` is None and (budget, sv_count) are provided, it chooses n so that n^2*sv_count <= budget.
      (Use sv_count = number of support vectors for SVMs / QSVMs.)
    - Saves to `save_path` and returns the Matplotlib Figure (does not display).
    """
    assert X.shape[1] == 2, "This helper assumes 2-D inputs."

    # Mesh size
    if n is None:
        if budget is None or sv_count is None:
            raise ValueError("Provide either a fixed n, or both budget and sv_count.")
        n = int(math.floor(math.sqrt(budget / max(1, sv_count))))
        n = max(min_n, min(max_n, n))

    # Limits
    if xlim is None or ylim is None:
        xlim = (X[:, 0].min() - pad, X[:, 0].max() + pad)
        ylim = (X[:, 1].min() - pad, X[:, 1].max() + pad)

    # Grid
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], n),
        np.linspace(ylim[0], ylim[1], n),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict (caller can batch inside predict_fn if needed)
    Z = predict_fn(grid).reshape(xx.shape)

    # Plot (no display)
    fig, ax = plt.subplots(figsize=(6, 5))
    levels = np.arange(Z.min() - 0.5, Z.max() + 1.5)
    ax.contourf(xx, yy, Z, levels=levels, alpha=0.25)
    for cls, color in [(0, "tab:blue"), (1, "tab:orange")]:
        m = (y == cls)
        ax.scatter(X[m, 0], X[m, 1], s=18, edgecolor="k", linewidth=0.3,
                   c=color, alpha=scatter_alpha, label=f"class {cls}")
    ax.set_xlim(xlim); ax.set_ylim(ylim)

    subtitle = f"(n={n})" if budget is None else f"(n={n}, SV={sv_count}, ~evals={n*n*sv_count:,} ≤ {budget:,})"
    ax.set_title(f"{title} {subtitle}", fontsize=10)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def plot_confusion_matrices_row(
    y_true: Sequence[int],
    y_preds: List[Sequence[int]],
    titles: List[str],
    save_path: str,
    *,
    labels: Optional[Sequence] = None,        # class order; defaults to sorted unique(y_true)
    normalize: Optional[str] = None,          # {None, 'true', 'pred', 'all'}
    cmap: str = "Blues",
    include_colorbar: bool = False,
    figsize: tuple = (16, 3),
    values_format: Optional[str] = None,      # auto: '.2f' if normalized else 'd'
) -> "matplotlib.figure.Figure":
    """
    Plot multiple confusion matrices in one horizontal row, save, and return the Figure.

    Args:
        y_true: Ground-truth labels for the shared test set.
        y_preds: List of prediction arrays, one per model.
        titles: List of panel titles (same length as y_preds).
        save_path: Output PNG path for the composite.
        labels: Class label order for the axes (optional).
        normalize: Normalization mode for confusion_matrix (None/'true'/'pred'/'all').
        cmap: Matplotlib colormap name.
        include_colorbar: If True, show a colorbar on the last panel.
        figsize: Figure size (width, height) in inches.
        values_format: Format string for cell values. Auto-chosen if None.

    Returns:
        matplotlib.figure.Figure
    """
    if len(y_preds) != len(titles):
        raise ValueError("y_preds and titles must have the same length.")

    y_true = np.asarray(y_true)
    if labels is None:
        # Stable class order derived from y_true
        labels = np.unique(y_true)

    n = len(y_preds)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    # choose values format
    if values_format is None:
        values_format = ".2f" if normalize in {"true", "pred", "all"} else "d"

    for i, (ax, yhat, title) in enumerate(zip(axes, y_preds, titles)):
        yhat = np.asarray(yhat)
        cm = confusion_matrix(y_true, yhat, labels=labels, normalize=normalize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        # only add colorbar to the last axis if requested
        disp.plot(ax=ax, cmap=cmap, colorbar=(include_colorbar and i == n - 1), values_format=values_format)
        ax.set_title(title)
        # make the row compact
        if i > 0:
            ax.set_ylabel("")  # avoid repeating ylabel
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

def compose_boundary_grid_from_files(
    image_paths: List[str],
    save_path: str,
    titles: Optional[List[str]] = None,
    cols: int = 2,
    dpi: int = 150,
    figsize_per_panel: float = 6.0,   # ~ width (inches) per column
) -> "matplotlib.figure.Figure":
    """
    Create a square-ish grid from already-saved boundary plot images.

    Args:
        image_paths: list of file paths to PNGs (created earlier).
        save_path: output PNG file for the composite.
        titles: optional titles per panel; if None, use empty strings.
        cols: number of columns in the grid (2 for 2x2).
        dpi: DPI for the composite figure.
        figsize_per_panel: width in inches per panel (tunes overall size).

    Returns:
        Composite matplotlib Figure (not displayed).
    """
    if not image_paths:
        raise ValueError("No image paths provided.")
    n = len(image_paths)
    rows = math.ceil(n / cols)

    # Read all images
    imgs = [plt.imread(p) for p in image_paths]
    if titles is None:
        titles = [""] * n
    if len(titles) != n:
        raise ValueError("Length of 'titles' must match number of images.")

    # Estimate composite size from first image aspect
    h0, w0 = imgs[0].shape[:2]
    panel_aspect = w0 / max(1, h0)  # width / height
    fig_w = cols * figsize_per_panel
    fig_h = rows * (figsize_per_panel / max(1e-9, panel_aspect))

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi)
    axes = np.array(axes).ravel() if isinstance(axes, np.ndarray) else np.array([axes])

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(imgs[i])
            ax.set_title(titles[i], fontsize=12, pad=8)
            ax.axis("off")
        else:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
