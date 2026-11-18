import numpy as np
from numpy.testing import assert_allclose

from bn_pah_fes.config import Parameters
from bn_pah_fes.fitting import FitResult
from bn_pah_fes.surfaces import (
    calculate_G0,
    calculate_G0_harmonic_analytic,
)


def _harmonic_fit_result() -> FitResult:
    theta = np.array([
        0.7, -0.2, 0.3, -0.4, 1.8,
        0.6, 0.9, 0.2, -0.15, 0.1,
    ])
    return FitResult(
        theta=theta,
        q_mean=np.array([0.4, -0.2, 0.1]),
        q_std=np.array([0.7, 0.5, 0.4]),
        q_scaled=np.empty((0, 3)),
        design_matrix=np.empty((0, 10)),
        delta_g_pbe=np.empty(0),
        optimization_result=None,
        optimization_history=[],
        free_energy_history=[],
    )


def test_finite_analytic_harmonic_projection_matches_quadrature() -> None:
    params = Parameters(temperature=1.0, kB=1.0, model="harmonic")
    fit_result = _harmonic_fit_result()
    qS = np.array([-0.1, 0.2])
    qT = np.array([0.05, -0.15])
    q0_grid = np.linspace(-0.3, 1.4, 20_001)

    numerical = calculate_G0(qS, qT, q0_grid, fit_result, params)
    analytic = calculate_G0_harmonic_analytic(
        qS, qT, fit_result, params,
        q0_bounds=(q0_grid[0], q0_grid[-1]),
    )

    assert_allclose(numerical, analytic, rtol=1.0e-8, atol=1.0e-10)


def test_full_line_harmonic_projection_is_the_limit_of_wide_bounds() -> None:
    params = Parameters(temperature=1.0, kB=1.0, model="harmonic")
    fit_result = _harmonic_fit_result()
    qS = np.array([-0.1, 0.2])
    qT = np.array([0.05, -0.15])

    full_line = calculate_G0_harmonic_analytic(qS, qT, fit_result, params)
    wide_bounds = calculate_G0_harmonic_analytic(
        qS, qT, fit_result, params, q0_bounds=(-5.0, 5.0)
    )

    assert_allclose(wide_bounds, full_line, rtol=1.0e-12, atol=1.0e-12)
