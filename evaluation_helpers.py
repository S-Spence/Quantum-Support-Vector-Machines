import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, List
import matplotlib as mpl
from matplotlib.colors import ListedColormap

def decision_boundary_plot(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    save_path: str,
    title: str = "Decision boundary",
    n: Optional[int] = None, # mesh size
    budget: Optional[int] = None, # cap ~ n^2 * sv_count
    sv_count: Optional[int] = None, # number of support vectors
    pad: float = 0.15,
    min_n: int = 35,
    max_n: int = 300,
    scatter_alpha: float = 0.9,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> "matplotlib.figure.Figure":
    """
    Plot decision boundary for 2-D data using a provided prediction function. The budget can be used to limit
    the number of kernel evaluations to speed up runtimes for the quantum svms. 

    Params:
        predict_fn: function that takes (M x 2) array and returns (M,) array of class predictions
        x: (N x 2) array of 2-D inputs
        y: (N,) array of class labels
        save_path: file path to save the resulting plot (PNG)
        title: title for the plot
        n: number of points per axis in the mesh grid; if None, computed from budget and sv_count
        budget: approximate max number of kernel evaluations (n^2 * sv_count); used if n is None
        sv_count: number of support vectors; used if n is None
        pad: padding added to data limits for the plot
        min_n: minimum n if computed from budget
        max_n: maximum n if computed from budget
        scatter_alpha: alpha value for scatter plot points
        xlim: optional x-axis limits (min, max); if None, computed from data
        ylim: optional y-axis limits (min, max); if None, computed from data
    """
    assert x.shape[1] == 2, "Input data x must be 2-D for decision boundary plot."

    # mesh size
    if n is None:
        if budget is None or sv_count is None:
            raise ValueError("Provide either a fixed n, or both budget and sv_count.")
        n = int(math.floor(math.sqrt(budget / max(1, sv_count))))
        n = max(min_n, min(max_n, n))

    # limits
    if xlim is None or ylim is None:
        xlim = (x[:, 0].min() - pad, x[:, 0].max() + pad)
        ylim = (x[:, 1].min() - pad, x[:, 1].max() + pad)

    # grid
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], n),
        np.linspace(ylim[0], ylim[1], n),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    z = predict_fn(grid).reshape(xx.shape)

    # lighten background colors
    bg_cmap = ListedColormap([
        lighten("tab:blue",   0.6),
        lighten("tab:orange", 0.6),
    ])

    # plot without display
    fig, ax = plt.subplots(figsize=(6, 5))
    levels = np.arange(z.min() - 0.5, z.max() + 1.5)
    ax.contourf(xx, yy, z, levels=levels, alpha=0.25, cmap=bg_cmap)
    for cls, color in [(0, "tab:blue"), (1, "tab:orange")]:
        m = (y == cls)
        ax.scatter(x[m, 0], x[m, 1], s=18, edgecolor="k", linewidth=0.3,
                   c=color, alpha=scatter_alpha, label=f"class {cls}")
    ax.set_xlim(xlim); ax.set_ylim(ylim)

    ax.set_title(f"{title}", fontsize=10)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def lighten(color, factor=0.4):
    r,g,b,a = mpl.colors.to_rgba(color)
    return (1 - (1-r)*factor, 1 - (1-g)*factor, 1 - (1-b)*factor, 1.0)

def compose_boundary_grid_from_files(
    image_paths: List[str],
    save_path: str,
    titles: Optional[List[str]] = None,
    cols: int = 2,
    figsize_per_panel: float = 6.0,
) -> "matplotlib.figure.Figure":
    """
    Create a square grid from saved boundary plot images.

    Params:
        image_paths: list of file paths to boundary plot images
        save_path: output file path for the composite image (PNG)
        titles: optional titles per panel
        cols: number of columns in the grid
        figsize_per_panel: width in inches per panel (tunes overall size)

    Returns:
        Composite matplotlib Figure (not displayed).
    """
    if not image_paths:
        raise ValueError("No image paths provided.")
    n = len(image_paths)
    rows = math.ceil(n / cols)

    # read all images
    imgs = [plt.imread(p) for p in image_paths]
    if titles is None:
        titles = [""] * n
    if len(titles) != n:
        raise ValueError("Length of 'titles' must match number of images.")

    # estimate composite size from first image aspect
    h0, w0 = imgs[0].shape[:2]
    panel_aspect = w0 / max(1, h0)
    fig_w = cols * figsize_per_panel
    fig_h = rows * (figsize_per_panel / max(1e-9, panel_aspect))

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=150)
    axes = np.array(axes).ravel() if isinstance(axes, np.ndarray) else np.array([axes])
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(imgs[i])
            ax.set_title(titles[i], fontsize=12, pad=8)
            ax.axis("off")
        else:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
