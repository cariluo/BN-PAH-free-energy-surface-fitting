from dataclasses import dataclass

import numpy as np

from .kde import KDEResult
from .surfaces import SurfaceResult


@dataclass(frozen=True)
class MLEKDEComparison:
    """Quantitative comparison between the MLE and KDE G0 surfaces."""

    grid_rmse: float
    grid_mae: float
    grid_max_abs: float
    sampled_rmse: float
    sampled_mae: float
    sampled_max_abs: float
    grid_difference: np.ndarray


def compare_mle_to_kde(
    surface_result: SurfaceResult,
    kde_result: KDEResult,
) -> MLEKDEComparison:
    """Compare normalized MLE and KDE G0 surfaces on the same qS/qT grid."""
    mle_grid = surface_result.G0_surface
    kde_grid = kde_result.free_energy.T

    if mle_grid.shape != kde_grid.shape:
        raise ValueError(
            "MLE and KDE grids must have the same shape; "
            f"got {mle_grid.shape} and {kde_grid.shape}."
        )

    grid_difference = mle_grid - kde_grid
    sampled_difference = surface_result.G0_sampled - kde_result.sampled_free_energy

    return MLEKDEComparison(
        grid_rmse=float(np.sqrt(np.mean(grid_difference**2))),
        grid_mae=float(np.mean(np.abs(grid_difference))),
        grid_max_abs=float(np.max(np.abs(grid_difference))),
        sampled_rmse=float(np.sqrt(np.mean(sampled_difference**2))),
        sampled_mae=float(np.mean(np.abs(sampled_difference))),
        sampled_max_abs=float(np.max(np.abs(sampled_difference))),
        grid_difference=grid_difference,
    )
