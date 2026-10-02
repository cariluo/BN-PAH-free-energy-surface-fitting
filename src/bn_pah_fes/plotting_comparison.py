from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import EnergyData
from .diagnostics import MLEKDEComparison
from .kde import KDEResult

plt.rcParams["font.family"] = "Times New Roman"


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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    for ax, surface, title in zip(
        axes[:2],
        [mle_surface, kde_surface],
        [r"MLE $Delta G_0$", r"KDE $Delta G_0$"],
    ):
        contour = ax.contourf(qS_grid, qT_grid, surface, levels=30, cmap="viridis")
        ax.set_xlabel(r"$q_S$ (Ha)", fontsize=20)
        ax.set_ylabel(r"$q_T$ (Ha)", fontsize=20)
        ax.set_title(title, fontsize=22)
        ax.tick_params(labelsize=14)
        fig.colorbar(contour, ax=ax, label="Free energy (Ha)")

    limit = np.max(np.abs(difference))
    contour = axes[2].contourf(
        qS_grid,
        qT_grid,
        difference,
        levels=30,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axes[2].set_xlabel(r"$q_S$ (Ha)", fontsize=20)
    axes[2].set_ylabel(r"$q_T$ (Ha)", fontsize=20)
    axes[2].set_title(r"MLE $-$ KDE", fontsize=22)
    axes[2].tick_params(labelsize=14)
    fig.colorbar(contour, ax=axes[2], label="Difference (Ha)")

    fig.suptitle(
        f"MLE vs KDE: RMSE = {comparison.grid_rmse:.3e} Ha",
        fontsize=24,
    )
    fig.savefig(results_dir / "MLE_vs_KDE.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
