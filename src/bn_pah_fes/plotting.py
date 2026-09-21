from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import EnergyData
from .kde import KDEResult


plt.rcParams["font.family"] = "Times New Roman"


# Shared 3D surface plotting parameters.
SURFACE_FIGSIZE = (9, 7)
SURFACE_VIEW = (30, -60)
SURFACE_CMAP = "viridis"
SURFACE_ALPHA = 0.7
SURFACE_EDGE_COLOR = "none"
SAMPLE_SIZE = 10
SAMPLE_FACE_COLOR = "white"
SAMPLE_EDGE_COLOR = "black"
SAMPLE_ALPHA = 0.8
AXIS_LABEL_FONTSIZE = 30
AXIS_TICK_FONTSIZE = 15
AXIS_LABEL_PAD = 15
COLORBAR_SHRINK = 0.7
COLORBAR_PAD = 0.1


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
    fig = plt.figure(figsize=SURFACE_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=SURFACE_VIEW[0], azim=SURFACE_VIEW[1])
    surface = ax.plot_surface(
        kde_result.QS,
        kde_result.QT,
        kde_result.free_energy,
        cmap=SURFACE_CMAP,
        edgecolor=SURFACE_EDGE_COLOR,
        alpha=SURFACE_ALPHA,
    )
    ax.scatter(
        data.qS,
        data.qT,
        kde_result.sampled_free_energy + 0.01,
        s=SAMPLE_SIZE,
        facecolor=SAMPLE_FACE_COLOR,
        edgecolor=SAMPLE_EDGE_COLOR,
        alpha=SAMPLE_ALPHA,
    )
    ax.set_xlabel(
        r"$q_S$ (Ha)",
        fontsize=AXIS_LABEL_FONTSIZE,
        labelpad=AXIS_LABEL_PAD,
    )
    ax.set_ylabel(
        r"$q_T$ (Ha)",
        fontsize=AXIS_LABEL_FONTSIZE,
        labelpad=AXIS_LABEL_PAD,
    )
    ax.set_zlabel(
        r"$-k_BT\ln P(q_S,q_T)$ (Ha)",
        fontsize=AXIS_LABEL_FONTSIZE,
        labelpad=AXIS_LABEL_PAD,
    )
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    ax.tick_params(axis="z", labelsize=AXIS_TICK_FONTSIZE)
    fig.colorbar(
        surface,
        ax=ax,
        shrink=COLORBAR_SHRINK,
        pad=COLORBAR_PAD,
        #label=r"$-k_BT\ln P(q_S,q_T)$ (Ha)",
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
    minimum: tuple[float, float, float] | None = None,
) -> None:
    """Plot a 3D free-energy surface and sampled trajectory points."""
    fig = plt.figure(figsize=SURFACE_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=SURFACE_VIEW[0], azim=SURFACE_VIEW[1])
    surface = ax.plot_surface(
        QS,
        QT,
        surface_data,
        cmap=SURFACE_CMAP,
        edgecolor=SURFACE_EDGE_COLOR,
        alpha=SURFACE_ALPHA,
    )
    ax.scatter(
        q[:, 1],
        q[:, 2],
        sampled,
        s=SAMPLE_SIZE,
        facecolor=SAMPLE_FACE_COLOR,
        edgecolor=SAMPLE_EDGE_COLOR,
        alpha=SAMPLE_ALPHA,
    )
    if minimum is not None:
        qS_minimum, qT_minimum, G_minimum = minimum
        ax.scatter(
            qS_minimum,
            qT_minimum,
            G_minimum,
            marker="*",
            s=180,
            facecolor="red",
            edgecolor="black",
            linewidth=1.0,
            label=f"Minimum: (qS, qT) = ({qS_minimum:.4f}, {qT_minimum:.4f}) Ha",
        )
        ax.legend(loc="best", fontsize=20)
    ax.set_xlabel(
        r"$q_S$ (Ha)",
        fontsize=AXIS_LABEL_FONTSIZE,
        labelpad=AXIS_LABEL_PAD,
    )
    ax.set_ylabel(
        r"$q_T$ (Ha)",
        fontsize=AXIS_LABEL_FONTSIZE,
        labelpad=AXIS_LABEL_PAD,
    )
    ax.set_zlabel(
        zlabel,
        fontsize=AXIS_LABEL_FONTSIZE,
        labelpad=AXIS_LABEL_PAD,
    )
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    ax.tick_params(axis="z", labelsize=AXIS_TICK_FONTSIZE)
    cbar = fig.colorbar(
        surface,
        ax=ax,
        shrink=COLORBAR_SHRINK,
        pad=COLORBAR_PAD,
    )
    #cbar.set_label(zlabel)
    cbar.ax.tick_params(labelsize=AXIS_TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(results_dir / filename, dpi=300)
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
    minimum: tuple[float, float, float] | None = None,
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
    if minimum is not None:
        qS_minimum, qT_minimum, _ = minimum
        plt.scatter(
            qS_minimum,
            qT_minimum,
            marker="*",
            s=220,
            facecolors="red",
            edgecolors="black",
            linewidths=1.0,
            label=f"Minimum: (qS, qT) = ({qS_minimum:.4f}, {qT_minimum:.4f}) Ha",
            zorder=5,
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
    plt.legend(loc="best", fontsize=20)
    plt.xlabel(r"$q_S$ (Ha)", fontsize=20)
    plt.ylabel(r"$q_T$ (Ha)", fontsize=20)
    plt.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    cbar = plt.colorbar(
        contour,
        ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    cbar.set_label(cbarlabel, fontsize=20)
    cbar.ax.tick_params(labelsize=AXIS_TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {results_dir / filename}")
