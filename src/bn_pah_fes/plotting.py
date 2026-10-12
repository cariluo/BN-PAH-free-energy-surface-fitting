from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

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
MINIMUM_ANNOTATION_FONTSIZE = 30
HARTREE_TO_EV = 27.211386245988


def _to_ev(values: np.ndarray | float | list[float]) -> np.ndarray:
    """Convert scalar or array-like Hartree values to electronvolts for plotting."""
    return np.asarray(values) * HARTREE_TO_EV


def _minimum_is_in_grid(
    minimum: tuple[float, float, float],
    QS: np.ndarray,
    QT: np.ndarray,
) -> bool:
    """Return whether a continuous minimum lies within the surface grid."""
    qS_minimum, qT_minimum, _ = minimum
    return (
        QS.min() <= qS_minimum <= QS.max()
        and QT.min(def _bilinear_interpolate(
    qS_grid: np.ndarray,
    qT_grid: np.ndarray,
    surface_data: np.ndarray,
    qS_values: np.ndarray,
    qT_values: np.ndarray,
) -> np.ndarray:
    """Interpolate a surface at matching qS and qT coordinates."""
    qS_values = np.asarray(qS_values)
    qT_values = np.asarray(qT_values)
    if qS_values.shape != qT_values.shape:
        raise ValueError("qS_values and qT_values must have the same shape.")
    if (
        np.any(qS_values < qS_grid.min())
        or np.any(qS_values > qS_grid.max())
        or np.any(qT_values < qT_grid.min())
        or np.any(qT_values > qT_grid.max())
    ):
        raise ValueError("Interpolation coordinates must lie within the surface grid.")

    qS_indices = np.clip(
        np.searchsorted(qS_grid, qS_values, side="left"),
        1,
        len(qS_grid) - 1,
    )
    qT_indices = np.clip(
        np.searchsorted(qT_grid, qT_values, side="left"),
        1,
        len(qT_grid) - 1,
    )
    qS_lower = qS_grid[qS_indices - 1]
    qS_upper = qS_grid[qS_indices]
    qT_lower = qT_grid[qT_indices - 1]
    qT_upper = qT_grid[qT_indices]
    qS_weight = (qS_values - qS_lower) / (qS_upper - qS_lower)
    qT_weight = (qT_values - qT_lower) / (qT_upper - qT_lower)

    lower_values = (
        (1.0 - qT_weight) * surface_data[qS_indices - 1, qT_indices - 1]
        + qT_weight * surface_data[qS_indices - 1, qT_indices]
    )
    upper_values = (
        (1.0 - qT_weight) * surface_data[qS_indices, qT_indices - 1]
        + qT_weight * surface_data[qS_indices, qT_indices]
    )
    return (1.0 - qS_weight) * lower_values + qS_weight * upper_values


def _project_gap_profile(
    qS_grid: np.ndarray,
    qT_grid: np.ndarray,
    surface_data: np.ndarray,
    beta: float,
    n_gap_points: int | None = None,
    n_integration_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a 2D surface onto q=qS-qT by integrating out qbar."""
    gap_min = qS_grid.min() - qT_grid.max()
    gap_max = qS_grid.max() - qT_grid.min()
    n_gap_points = n_gap_points or max(len(qS_grid), len(qT_grid))
    n_integration_points = n_integration_points or max(
        len(qS_grid), len(qT_grid)
    )
    # Omit the endpoints, where the allowed qbar interval has zero width.
    gap_grid = np.linspace(gap_min, gap_max, n_gap_points + 2)[1:-1]
    projected_free_energy = np.empty_like(gap_grid)

    for index, gap in enumerate(gap_grid):
        qbar_min = max(qS_grid.min() - 0.5 * gap, qT_grid.min() + 0.5 * gap)
        qbar_max = min(qS_grid.max() - 0.5 * gap, qT_grid.max() + 0.5 * gap)
        qbar_grid = np.linspace(qbar_min, qbar_max, n_integration_points)
        surface_values = _bilinear_interpolate(
            qS_grid,
            qT_grid,
            surface_data,
            qbar_grid + 0.5 * gap,
            qbar_grid - 0.5 * gap,
        )
        log_weights = np.full(
            n_integration_points,
            np.log((qbar_max - qbar_min) / (n_integration_points - 1)),
        )
        log_weights[[0, -1]] += np.log(0.5)
        projected_free_energy[index] = -logsumexp(
            -beta * surface_values + log_weights
        ) / beta

    return gap_grid, projected_free_energy


def plot_singlet_triplet_gap_profiles(
    qS_grid: np.ndarray,
    qT_grid: np.ndarray,
    GS_surface: np.ndarray,
    GT_surface: np.ndarray,
    beta: float,
    results_dir: Path,
) -> None:
    """Plot singlet and triplet free energies projected onto qS-qT."""
    q_gap, GS_gap = _project_gap_profile(
        qS_grid, qT_grid, GS_surface, beta
    )
    _, GT_gap = _project_gap_profile(qS_grid, qT_grid, GT_surface, beta)

    plt.figure(figsize=(8, 6))
    plt.plot(
        _to_ev(q_gap),
        _to_ev(GS_gap),
        linewidth=2,
        label=r"$\Delta G_S$",
    )
    plt.plot(
        _to_ev(q_gap),
        _to_ev(GT_gap),
        linestyle="--",
        linewidth=2,
        label=r"$\Delta G_T$",
    )
    plt.xlabel(r"$q=q_S-q_T$ (eV)", fontsize=30)
    plt.ylabel(r"$\Delta G$ (eV)", fontsize=30)
    plt.tick_params(axis="both", labelsize=18)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(
        results_dir / "singlet_triplet_gap_profiles.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved {results_dir / 'singlet_triplet_gap_profiles.png'}")


def plot_coordinate_time_series(
    data: EnergyData,
    results_dir: Path,
) -> None:
    """Plot all and subsampled q0, qS, and qT trajectories."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].scatter(data.idx_all, _to_ev(data.q0_all), s=2, label="All")
    axes[0].scatter(data.idx, _to_ev(data.q0), s=16, label="Subsampled")
    axes[0].set_ylabel(r"$q_0$ (eV)", fontsize=30)
    axes[0].legend(fontsize=10)
    axes[1].scatter(data.idx_all, _to_ev(data.qS_all), s=2)
    axes[1].scatter(data.idx, _to_ev(data.qS), s=16)
    axes[1].set_ylabel(r"$q_S$ (eV)", fontsize=30)
    axes[2].scatter(data.idx_all, _to_ev(data.qT_all), s=2)
    axes[2].scatter(data.idx, _to_ev(data.qT), s=16)
    axes[2].set_ylabel(r"$q_T$ (eV)", fontsize=30)
    plt.tight_layout()
    plt.savefig(results_dir / "coordinate_time_series.png", dpi=300)
    plt.close()


def plot_energy_correlation(
    data: EnergyData,
    results_dir: Path,
) -> None:
    """Plot the correlation between q0 and qS coordinates."""
    plt.figure(figsize=(8, 6))
    plt.scatter(
        _to_ev(data.q0),
        _to_ev(data.qS),
        s=20,
        facecolors="white",
        edgecolors="black",
        alpha=0.8,
    )

    slope, intercept = np.polyfit(data.q0, data.qS, 1)
    x_fit = np.linspace(data.q0.min(), data.q0.max(), 200)
    plt.plot(
        _to_ev(x_fit),
        _to_ev(slope * x_fit + intercept),
        linewidth=2,
        label=fr"$r = {np.corrcoef(data.q0, data.qS)[0, 1]:.4f}$",
    )

    plt.xlabel(r"$q_0$ (eV)", fontsize=30)
    plt.ylabel(r"$q_S$ (eV)", fontsize=30)
    plt.tick_params(axis="both", labelsize=18)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(results_dir / "q0_qS_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()



def plot_singlet_triplet_gap_histogram(
    data: EnergyData,
    results_dir: Path,
    beta: float,
) -> None:
    """Plot a q0-reweighted histogram of the singlet-triplet gaps qS - qT."""
    singlet_triplet_gap = _to_ev(data.qS - data.qT)
    log_weights = -beta * data.q0
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    weights /= weights.sum()

    plt.figure(figsize=(8, 6))
    bins = np.histogram_bin_edges(singlet_triplet_gap, bins=30)
    negative_mask = singlet_triplet_gap < 0
    negative_gap = singlet_triplet_gap[negative_mask]
    nonnegative_gap = singlet_triplet_gap[~negative_mask]

    negative_counts, _ = np.histogram(
        negative_gap,
        bins=bins,
        weights=weights[negative_mask],
    )
    nonnegative_counts, _ = np.histogram(
        nonnegative_gap,
        bins=bins,
        weights=weights[~negative_mask],
    )

    plt.bar(
        bins[:-1],
        negative_counts,
        width=np.diff(bins),
        align="edge",
        color="green",
        edgecolor="black",
        alpha=0.8,
    )
    plt.bar(
        bins[:-1],
        nonnegative_counts,
        width=np.diff(bins),
        align="edge",
        edgecolor="black",
        alpha=0.8,
    )

    counts = negative_counts + nonnegative_counts
    mean = np.average(singlet_triplet_gap, weights=weights)
    variance = np.average((singlet_triplet_gap - mean) ** 2, weights=weights)
    std = np.sqrt(variance)

    x_fit = np.linspace(bins[0], bins[-1], 300)
    gaussian = counts.max() * np.exp(-0.5 * ((x_fit - mean) / std) ** 2)
    plt.plot(
        x_fit,
        gaussian,
        linewidth=2,
        label=f"$\\mu = {mean:.4f}$ eV\n$\\sigma = {std:.4f}$ eV",
    )

    plt.xlabel(r"$q_S-q_T$ (eV)", fontsize=30)
    plt.ylabel("Reweighted probability", fontsize=30)
    plt.tick_params(axis="both", labelsize=18)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(
        results_dir / "singlet_triplet_gap_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def plot_fit_free_energy_history(
    history: list[float],
    results_dir: Path,
    model: str,
) -> None:
    """Plot mean sampled Delta G_PBE over L-BFGS-B fit iterations."""
    if not history:
        return

    iterations = np.arange(len(history))
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, _to_ev(np.asarray(history)), linewidth=2)
    plt.scatter(iterations, _to_ev(np.asarray(history)), s=35, zorder=3)
    plt.xlabel("Optimization iteration", fontsize=30)
    plt.ylabel(r"Mean $\Delta G_{\mathrm{PBE}}$ (eV)", fontsize=24)
    plt.title(f"{model.replace('_', ' ').title()} model", fontsize=24)
    plt.tick_params(axis="both", labelsize=18)
    plt.tight_layout()
    plt.savefig(
        results_dir / f"free_energy_history_{model}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_likelihood_convergence(
    history: list[float],
    final_objective: float,
    results_dir: Path,
    model: str,
) -> None:
    """Plot positive likelihood gaps relative to the final objective."""
    iterations = np.arange(len(history))
    objective_gap = np.asarray(history) - final_objective
    positive_gap = objective_gap > 0.0
    if not np.any(positive_gap):
        return

    plt.figure(figsize=(8, 6))
    plt.semilogy(
        iterations[positive_gap],
        objective_gap[positive_gap],
        linewidth=2,
        marker="o",
        markersize=5,
    )
    plt.xlabel("Optimization iteration", fontsize=30)
    plt.ylabel(
        r"$\mathcal{L} - \mathcal{L}_{\mathrm{final}}$",
        fontsize=24,
    )
    plt.title(f"{model.replace('_', ' ').title()} model", fontsize=24)
    plt.tick_params(axis="both", labelsize=18)
    plt.tight_layout()
    plt.savefig(
        results_dir / f"likelihood_convergence_{model}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_kde_surface(
    data: EnergyData,
    kde_result: KDEResult,
    results_dir: Path,
    surface_limits: tuple[float, float] | None = None,
) -> None:
    """Plot the weighted empirical 2D free-energy surface."""
    fig = plt.figure(figsize=SURFACE_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=SURFACE_VIEW[0], azim=SURFACE_VIEW[1])
    surface = ax.plot_surface(
        _to_ev(kde_result.QS),
        _to_ev(kde_result.QT),
        _to_ev(kde_result.free_energy),
        cmap=SURFACE_CMAP,
        edgecolor=SURFACE_EDGE_COLOR,
        alpha=SURFACE_ALPHA,
        vmin=None if surface_limits is None else _to_ev(surface_limits[0]),
        vmax=None if surface_limits is None else _to_ev(surface_limits[1]),
    )
    ax.scatter(
        _to_ev(data.qS),
        _to_ev(data.qT),
        _to_ev(kde_result.sampled_free_energy + 0.01),
        s=SAMPLE_SIZE,
        facecolor=SAMPLE_FACE_COLOR,
        edgecolor=SAMPLE_EDGE_COLOR,
        alpha=SAMPLE_ALPHA,
    )
    ax.set_xlabel(
        r"$q_S$ (eV)",
        fontsize=AXIS_LABEL_FONTSIZE,
        labelpad=AXIS_LABEL_PAD,
    )
    ax.set_ylabel(
        r"$q_T$ (eV)",
        fontsize=AXIS_LABEL_FONTSIZE,
        labelpad=AXIS_LABEL_PAD,
    )
    ax.set_zlabel(
        r"$-k_BT\ln P(q_S,q_T)$ (eV)",
        fontsize=AXIS_LABEL_FONTSIZE,
        labelpad=AXIS_LABEL_PAD,
    )
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    ax.tick_params(axis="z", labelsize=AXIS_TICK_FONTSIZE)
    if surface_limits is not None:
        ax.set_zlim(*_to_ev(np.asarray(surface_limits)))
    fig.colorbar(
        surface,
        ax=ax,
        shrink=COLORBAR_SHRINK,
        pad=COLORBAR_PAD,
        #label=r"$-k_BT\ln P(q_S,q_T)$ (eV)",
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
    surface_limits: tuple[float, float] | None = None,
) -> None:
    """Plot a 3D free-energy surface and sampled trajectory points."""
    fig = plt.figure(figsize=SURFACE_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=SURFACE_VIEW[0], azim=SURFACE_VIEW[1])
    surface = ax.plot_surface(
        _to_ev(QS),
        _to_ev(QT),
        _to_ev(surface_data),
        cmap=SURFACE_CMAP,
        edgecolor=SURFACE_EDGE_COLOR,
        alpha=SURFACE_ALPHA,
        vmin=None if surface_limits is None else _to_ev(surface_limits[0]),
        vmax=None if surface_limits is None else _to_ev(surface_limits[1]),
    )
    ax.scatter(
        _to_ev(q[:, 1]),
        _to_ev(q[:, 2]),
        _to_ev(sampled),
        s=SAMPLE_SIZE,
        facecolor=SAMPLE_FACE_COLOR,
        edgecolor=SAMPLE_EDGE_COLOR,
        alpha=SAMPLE_ALPHA,
    )
    z_limits = (
        surface_limits
        if surface_limits is not None
        else (float(surface_data.min()), float(surface_data.max()))
    )
    if minimum is not None and _minimum_is_in_grid(minimum, QS, QT):
        z_limits = (
            min(z_limits[0], minimum[2]),
            max(z_limits[1], minimum[2]),
        )

    ax.set_xlim(*_to_ev([QS.min(), QS.max()]))
    ax.set_ylim(*_to_ev([QT.min(), QT.max()]))
    ax.set_zlim(*_to_ev(z_limits))

    if minimum is not None:
        qS_minimum, qT_minimum, G_minimum = minimum
        minimum_label = (
            fr"Min: ($q_S$, $q_T$) = "
            fr"({_to_ev(qS_minimum):.4f}, {_to_ev(qT_minimum):.4f}) eV"
        )
        ax.text2D(
            0.02,
            0.98,
            minimum_label,
            transform=ax.transAxes,
            va="top",
            fontsize=MINIMUM_ANNOTATION_FONTSIZE,
        )
        if _minimum_is_in_grid(minimum, QS, QT):
            ax.scatter(
                _to_ev(qS_minimum),
                _to_ev(qT_minimum),
                _to_ev(G_minimum),
                marker="*",
                s=180,
                facecolor="red",
                edgecolor="black",
                linewidth=1.0,
            )
    ax.set_xlabel(
        r"$q_S$ (eV)",
        fontsize=AXIS_LABEL_FONTSIZE,
        labelpad=AXIS_LABEL_PAD,
    )
    ax.set_ylabel(
        r"$q_T$ (eV)",
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
        _to_ev(QS),
        _to_ev(QT),
        _to_ev(surface_data),
        levels=30,
        vmin=0.0,
        cmap="viridis",
    )
    plt.scatter(
        _to_ev(q[:, 1]),
        _to_ev(q[:, 2]),
        facecolors="white",
        edgecolors="black",
        label=f"{samples_in_fit} sampled points",
    )
    plt.xlim(*_to_ev([QS.min(), QS.max()]))
    plt.ylim(*_to_ev([QT.min(), QT.max()]))

    if minimum is not None:
        qS_minimum, qT_minimum, _ = minimum
        minimum_label = (
            fr"Continuous minimum: ($q_S$, $q_T$) = "
            fr"({_to_ev(qS_minimum):.4f}, {_to_ev(qT_minimum):.4f}) eV"
        )
        plt.gca().text(
            0.02,
            0.98,
            minimum_label,
            transform=plt.gca().transAxes,
            va="top",
            fontsize=MINIMUM_ANNOTATION_FONTSIZE,
        )
        if _minimum_is_in_grid(minimum, QS, QT):
            plt.scatter(
                _to_ev(qS_minimum),
                _to_ev(qT_minimum),
                marker="*",
                s=220,
                facecolors="red",
                edgecolors="black",
                linewidths=1.0,
                zorder=5,
            )
    diag_min = max(QS.min(), QT.min())
    diag_max = min(QS.max(), QT.max())
    plt.plot(
        _to_ev([diag_min, diag_max]),
        _to_ev([diag_min, diag_max]),
        color="white",
        linestyle="--",
        linewidth=2,
        label=r"$q_T=q_S$",
    )
    plt.legend(loc="best", fontsize=20)
    plt.xlabel(r"$q_S$ (eV)", fontsize=20)
    plt.ylabel(r"$q_T$ (eV)", fontsize=20)
    plt.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    cbar = plt.colorbar(
        contour,
        ticks=_to_ev([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    )
    cbar.set_label(cbarlabel, fontsize=20)
    cbar.ax.tick_params(labelsize=AXIS_TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {results_dir / filename}")
