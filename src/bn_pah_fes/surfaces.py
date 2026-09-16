from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .config import Parameters
from .fitting import FitResult
from .polynomial import free_energy, polynomial_basis


@dataclass(frozen=True)
class SurfaceResult:
    """Reweighted two-dimensional free-energy surfaces."""

    qS_grid: np.ndarray
    qT_grid: np.ndarray
    q0_grid: np.ndarray
    QS: np.ndarray
    QT: np.ndarray
    G0_surface: np.ndarray
    GS_surface: np.ndarray
    GT_surface: np.ndarray
    G0_sampled: np.ndarray
    GS_sampled: np.ndarray
    GT_sampled: np.ndarray
    G0_min: float


def calculate_G0(
    qS_values: np.ndarray,
    qT_values: np.ndarray,
    q0_grid: np.ndarray,
    fit_result: FitResult,
    params: Parameters,
) -> np.ndarray:
    """Calculate unshifted G0(qS, qT) by numerical integration over q0."""
    qS_values = np.asarray(qS_values)
    qT_values = np.asarray(qT_values)
    if qS_values.shape != qT_values.shape:
        raise ValueError("qS_values and qT_values must have the same shape.")

    dq0 = q0_grid[1] - q0_grid[0]
    G0_values = np.empty(qS_values.size)

    for index, (qS_value, qT_value) in enumerate(
        zip(qS_values.ravel(), qT_values.ravel())
    ):
        points = np.column_stack([
            q0_grid,
            np.full(len(q0_grid), qS_value),
            np.full(len(q0_grid), qT_value),
        ])
        points_scaled = (points - fit_result.q_mean) / fit_result.q_std
        G_PBE = free_energy(
            fit_result.theta,
            polynomial_basis(points_scaled),
        )
        exponent = -params.beta * (G_PBE + q0_grid)
        log_integral = logsumexp(exponent) + np.log(dq0)
        G0_values[index] = -params.kBT * log_integral

    return G0_values.reshape(qS_values.shape)


def calculate_surfaces(
    fit_result: FitResult,
    q: np.ndarray,
    params: Parameters,
) -> SurfaceResult:
    """Construct G0, GS, and GT surfaces and sampled values."""
    qS_min, qS_max = q[:, 1].min(), q[:, 1].max()
    qT_min, qT_max = q[:, 2].min(), q[:, 2].max()
    qS_pad = params.plot_padding_factor * (qS_max - qS_min)
    qT_pad = params.plot_padding_factor * (qT_max - qT_min)
    qS_grid = np.linspace(qS_min - qS_pad, qS_max, params.n_grid_surface)
    qT_grid = np.linspace(qT_min - qT_pad, qT_max, params.n_grid_surface)

    q0_min, q0_max = q[:, 0].min(), q[:, 0].max()
    q0_pad = params.q0_padding_factor * (q0_max - q0_min)
    q0_grid = np.linspace(q0_min - q0_pad, q0_max, params.n_q0)

    QS, QT = np.meshgrid(qS_grid, qT_grid, indexing="ij")
    G0_surface = calculate_G0(QS, QT, q0_grid, fit_result, params)

    GS_surface = G0_surface + QS
    GT_surface = G0_surface + QT
    G0_min = float(np.min(G0_surface))
    G0_surface -= G0_min
    GS_surface -= G0_min
    GT_surface -= G0_min

    G0_sampled = calculate_G0(q[:, 1], q[:, 2], q0_grid, fit_result, params)
    GS_sampled = G0_sampled + q[:, 1]
    GT_sampled = G0_sampled + q[:, 2]
    G0_sampled -= G0_min
    GS_sampled -= G0_min
    GT_sampled -= G0_min

    return SurfaceResult(
        qS_grid=qS_grid,
        qT_grid=qT_grid,
        q0_grid=q0_grid,
        QS=QS,
        QT=QT,
        G0_surface=G0_surface,
        GS_surface=GS_surface,
        GT_surface=GT_surface,
        G0_sampled=G0_sampled,
        GS_sampled=GS_sampled,
        GT_sampled=GT_sampled,
        G0_min=G0_min,
    )
