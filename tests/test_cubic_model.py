import numpy as np
from numpy.testing import assert_allclose

from bn_pah_fes.config import Parameters
from bn_pah_fes.fitting import _free_energy_parameter_gradient
from bn_pah_fes.polynomial import (
    polynomial,
    polynomial_basis,
    polynomial_gradient,
)
from bn_pah_fes.surfaces import _model_functions


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
