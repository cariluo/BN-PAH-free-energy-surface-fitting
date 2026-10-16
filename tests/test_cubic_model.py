import numpy as np
from numpy.testing import assert_allclose

from bn_pah_fes.config import Parameters
from bn_pah_fes.fitting import _free_energy_parameter_gradient
from bn_pah_fes.polynomial import (
    polynomial,
    polynomial_basis,
    polynomial_gradient,
)
from bn_pah_fes.plotting import _project_gap_profile
from bn_pah_fes.surfaces import _model_functions


def test_squared_quadratic_parameter_gradient_uses_the_quadratic_basis() -> None:
    """For G = P2², dG/dtheta is 2 P2 times the quadratic basis."""
    rng = np.random.default_rng(10)
    theta = rng.normal(size=10)
    X = rng.normal(size=(4, 10))

    assert_allclose(
        _free_energy_parameter_gradient(theta, X, "squared_quadratic"),
        2.0 * (X @ theta)[:, None] * X,
    )


def test_cubic_parameter_gradient_is_the_design_matrix() -> None:
    """For G = P3, dG/dtheta is the cubic polynomial basis."""
    rng = np.random.default_rng(0)
    theta = rng.normal(size=20)
    X = rng.normal(size=(4, 20))

    assert_allclose(
        _free_energy_parameter_gradient(theta, X, "cubic"),
        X,
    )


def test_cubic_coordinate_gradient_matches_finite_difference() -> None:
    """The coordinate derivative used by cubic surface minimization is exact."""
    rng = np.random.default_rng(1)
    theta = rng.normal(size=20)
    point = rng.normal(size=(1, 3))
    step = 1.0e-6

    analytic = polynomial_gradient(theta, point)[0]
    numerical = np.empty(3)
    for coordinate in range(3):
        displacement = np.zeros_like(point)
        displacement[0, coordinate] = step
        forward = polynomial(theta, polynomial_basis(point + displacement))
        backward = polynomial(theta, polynomial_basis(point - displacement))
        numerical[coordinate] = (forward[0] - backward[0]) / (2.0 * step)

    assert_allclose(analytic, numerical, rtol=1.0e-6, atol=1.0e-6)


def test_cubic_surface_model_is_unsquared() -> None:
    """Surface integration dispatches the cubic model without squaring it."""
    rng = np.random.default_rng(2)
    theta = rng.normal(size=20)
    points = rng.normal(size=(3, 3))

    basis, evaluate_free_energy, gradient, is_squared = _model_functions(
        Parameters(model="cubic")
    )

    assert not is_squared
    assert_allclose(
        evaluate_free_energy(theta, basis(points)),
        polynomial(theta, polynomial_basis(points)),
    )
    assert_allclose(gradient(theta, points), polynomial_gradient(theta, points))


def test_projected_singlet_and_triplet_profiles_differ_by_gap() -> None:
    """Projection retains the qS-qT offset between singlet and triplet profiles."""
    qS_grid = np.linspace(-2.0, 2.0, 7)
    qT_grid = np.linspace(-2.0, 2.0, 7)
    G0_surface = np.zeros((len(qS_grid), len(qT_grid)))
    GS_surface = G0_surface + qS_grid[:, None]
    GT_surface = G0_surface + qT_grid[None, :]

    gap_grid, GS_profile = _project_gap_profile(
        qS_grid, qT_grid, GS_surface, beta=1.0, n_gap_points=5
    )
    _, GT_profile = _project_gap_profile(
        qS_grid, qT_grid, GT_surface, beta=1.0, n_gap_points=5
    )

    assert_allclose(GS_profile - GT_profile, gap_grid)
