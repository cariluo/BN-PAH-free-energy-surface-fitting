import numpy as np

from bn_pah_fes.fitting import _squared_quadratic_initial_theta
from bn_pah_fes.polynomial import (
    free_energy,
    harmonic_basis,
    polynomial_basis,
    squared_quadratic_plus_cubic_free_energy,
    squared_quadratic_plus_cubic_gradient,
)


def test_zero_cubic_correction_matches_squared_quadratic():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(7, 3))
    squared_quadratic_theta = rng.normal(size=10)
    theta = _squared_quadratic_initial_theta(squared_quadratic_theta)

    expected = free_energy(squared_quadratic_theta, harmonic_basis(q))
    actual = squared_quadratic_plus_cubic_free_energy(theta, polynomial_basis(q))

    np.testing.assert_allclose(actual, expected)


def test_squared_quadratic_initial_theta_pads_pure_cubic_terms_with_zero():
    squared_quadratic_theta = np.arange(10, dtype=float)

    theta = _squared_quadratic_initial_theta(squared_quadratic_theta)

    np.testing.assert_array_equal(theta[:10], squared_quadratic_theta)
    np.testing.assert_array_equal(theta[10:], np.zeros(10))


def test_squared_quadratic_plus_cubic_coordinate_gradient_matches_finite_difference():
    rng = np.random.default_rng(3)
    theta = rng.normal(scale=0.1, size=20)
    q = rng.normal(size=(4, 3))
    analytic = squared_quadratic_plus_cubic_gradient(theta, q)

    step = 1.0e-6
    numerical = np.empty_like(analytic)
    for coordinate in range(3):
        displacement = np.zeros(3)
        displacement[coordinate] = step
        forward = squared_quadratic_plus_cubic_free_energy(
            theta, polynomial_basis(q + displacement)
        )
        backward = squared_quadratic_plus_cubic_free_energy(
            theta, polynomial_basis(q - displacement)
        )
        numerical[:, coordinate] = (forward - backward) / (2.0 * step)

    np.testing.assert_allclose(analytic, numerical, rtol=1.0e-6, atol=1.0e-6)
