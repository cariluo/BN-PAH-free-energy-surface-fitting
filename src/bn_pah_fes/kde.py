from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde

from .config import Parameters
from .data import EnergyData


@dataclass(frozen=True)
class KDEResult:
    """Weighted empirical free-energy surface from the trajectory."""

    qS_grid: np.ndarray
    qT_grid: np.ndarray
    QS: np.ndarray
    QT: np.ndarray
    probability: np.ndarray
    free_energy: np.ndarray
    sampled_free_energy: np.ndarray


def calculate_kde_surface(data: EnergyData, params: Parameters) -> KDEResult:
    """Calculate the reweighted 2D KDE free-energy surface."""
    weights = np.exp(-params.beta * data.q0)
    weights /= weights.sum()

    kde = gaussian_kde(
        np.vstack([data.qS, data.qT]),
        weights=weights,
    )

    qS_grid = np.linspace(data.qS.min(), data.qS.max(), params.n_grid_surface)
    qT_grid = np.linspace(data.qT.min(), data.qT.max(), params.n_grid_surface)
    QS, QT = np.meshgrid(qS_grid, qT_grid)
    positions = np.vstack([QS.ravel(), QT.ravel()])

    probability = kde(positions).reshape(QS.shape)
    free_energy = -params.kBT * np.log(probability)
    free_energy -= free_energy.min()

    interpolator = RegularGridInterpolator(
        (qT_grid, qS_grid),
        free_energy,
        bounds_error=False,
        fill_value=None,
    )
    sampled_free_energy = interpolator(
        np.column_stack([data.qT, data.qS])
    )

    return KDEResult(
        qS_grid=qS_grid,
        qT_grid=qT_grid,
        QS=QS,
        QT=QT,
        probability=probability,
        free_energy=free_energy,
        sampled_free_energy=sampled_free_energy,
    )
