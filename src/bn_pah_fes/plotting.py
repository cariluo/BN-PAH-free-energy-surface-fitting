from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import EnergyData
from .kde import KDEResult


plt.rcParams["font.family"] = "Times New Roman"


def plot_coordinate_time_series(
    data: EnergyData,
    results_dir: Path,
) -> None:
    """Plot all and subsampled q0, qS, and qT trajectories."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].scatter(data.idx_all, data.q0_all, s=2, label="All")
    axes[0].scatter(data.idx, data.q0, s=16, label="Subsampled")
    axes[0].set_ylabel(r"$q_0$", fontsize=30)
    axes[0].legend(fontsize=10)
    axes[1].scatter(data.idx_all, data.qS_all, s=2)
    axes[1].scatter(data.idx, data.qS, s=16)
    axes[1].set_ylabel(r"$q_S$", fontsize=30)
    axes[2].scatter(data.idx_all, data.qT_all, s=2)
    axes[2].scatter(data.idx, data.qT, s=16)
    axes[2].set_ylabel(r"$q_T$", fontsize=30)
    plt.tight_layout()
    plt.savefig(results_dir / "coordinate_time_series.png", dpi=300)
    plt.close()


def plot_kde_surface(
    data: EnergyData,
    kde_result: KDEResult,
    results_dir: Path,
) -> None:
    """Plot the weighted empirical 2D free-energy surface."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        kde_result.QS,
        kde_result.QT,
        kde_result.free_energy,
        cmap="viridis",
        edgecolor="none",
        alpha=0.7,
    )
    ax.scatter(
        data.qS,
        data.qT,
        kde_result.sampled_free_energy + 0.01,
        s=5,
        facecolor="white",
        edgecolor="black",
        alpha=0.5,
    )
    ax.set_xlabel(r"$q_S$", fontsize=30)
    ax.set_ylabel(r"$q_T$", fontsize=30)
    ax.set_zlabel(r"$-k_BT\ln P(q_S,q_T)$ (Ha)")
    fig.colorbar(
        surface,
        ax=ax,
        shrink=0.7,
        pad=0.1,
        label=r"$-k_BT\ln P(q_S,q_T)$ (Ha)",
    )
    plt.tight_layout()
    plt.savefig(results_dir / "KDE.png", dpi=300)
    plt.close()


def plot_3d_free_energy_surface(
    QS: np.ndarray,
    QT: np.ndarray,
    surface_data: np.ndarray,
    q: np.ndarray,
    sampled: np.ndarray,
    filename: str,
    zlabel: str,
    results_dir: Path,
    samples_in_fit: int,
) -> None:
    """Plot a 3D free-energy surface and sampled trajectory points."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=30, azim=-60)
    surface = ax.plot_surface(
        QS,
        QT,
        surface_data,
        cmap="viridis",
        alpha=0.8,
    )
    ax.scatter(
        q[:, 1],
        q[:, 2],
        sampled,
        facecolors="white",
        edgecolors="black",
        label=f"{samples_in_fit} sampled points",
        depthshade=False,
    )
    ax.legend(loc="upper right")
    ax.set_xlabel(r"$q_S$ (Ha)", fontsize=20)
    ax.set_ylabel(r"$q_T$ (Ha)", fontsize=20)
    ax.set_zlabel(zlabel, fontsize=20)
    cbar = fig.colorbar(
        surface,
        ax=ax,
        shrink=0.7,
        pad=0.1,
        #ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    cbar.set_label(zlabel, fontsize=20)
    plt.tight_layout()
    plt.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {results_dir / filename}")


def plot_delta_g_contour(
    QS: np.ndarray,
    QT: np.ndarray,
    surface_data: np.ndarray,
    q: np.ndarray,
    filename: str,
    cbarlabel: str,
    results_dir: Path,
    samples_in_fit: int,
) -> None:
    """Plot a 2D free-energy contour with sampled points and qT=qS."""
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(
        QS,
        QT,
        surface_data,
        levels=30,
        vmin=0.0,
        vmax=0.65,
        cmap="viridis",
    )
    plt.scatter(
        q[:, 1],
        q[:, 2],
        facecolors="white",
        edgecolors="black",
        label=f"{samples_in_fit} sampled points",
    )
    diag_min = max(QS.min(), QT.min())
    diag_max = min(QS.max(), QT.max())
    plt.plot(
        [diag_min, diag_max],
        [diag_min, diag_max],
        color="white",
        linestyle="--",
        linewidth=2,
        label=r"$q_T=q_S$",
    )
    plt.legend(loc="best")
    plt.xlabel(r"$q_S$ (Ha)", fontsize=20)
    plt.ylabel(r"$q_T$ (Ha)", fontsize=20)
    cbar = plt.colorbar(
        contour,
        ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    cbar.set_label(cbarlabel, fontsize=20)
    plt.tight_layout()
    plt.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {results_dir / filename}")
