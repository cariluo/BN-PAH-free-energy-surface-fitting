from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import MLEKDEComparison
from .kde import KDEResult

plt.rcParams["font.family"] = "Times New Roman"

VIEW_ELEV = 30
VIEW_AZIM = -60


def plot_mle_kde_comparison(
    qS_grid: np.ndarray,
    qT_grid: np.ndarray,
    mle_surface: np.ndarray,
    kde_result: KDEResult,
    comparison: MLEKDEComparison,
    results_dir: Path,
) -> None:
    """Plot MLE and KDE G0 surfaces and their pointwise difference."""
    kde_surface = kde_result.free_energy.T
    difference = comparison.grid_difference

    fig = plt.figure(figsize=(18, 6))

    surfaces = [
        (mle_surface, r"MLE $\Delta G_0$", "viridis"),
        (kde_surface, r"KDE $\Delta G_0$", "viridis"),
        (difference, r"MLE $-$ KDE", "coolwarm"),
    ]

    for i, (surface, title, cmap) in enumerate(surfaces, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        plotted = ax.plot_surface(
            qS_grid,
            qT_grid,
            surface,
            cmap=cmap,
            edgecolor="none",
            alpha=0.85,
        )
        ax.set_xlabel(r"$q_S$ (Ha)", fontsize=16, labelpad=8)
        ax.set_ylabel(r"$q_T$ (Ha)", fontsize=16, labelpad=8)
        ax.set_zlabel(
            "Free energy (Ha)" if i < 3 else "Difference (Ha)",
            fontsize=16,
            labelpad=8,
        )
        ax.set_title(title, fontsize=20)
        ax.tick_params(axis="both", labelsize=11)
        ax.tick_params(axis="z", labelsize=11)

        # Use the same orthographic camera for every panel.
        ax.set_proj_type("ortho")
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

        fig.colorbar(plotted, ax=ax, shrink=0.65, pad=0.08)

    fig.suptitle(
        f"MLE vs KDE: RMSE = {comparison.grid_rmse:.3e} Ha",
        fontsize=24,
    )
    fig.savefig(results_dir / "MLE_vs_KDE.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
