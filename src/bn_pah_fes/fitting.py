from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .config import Parameters
from .data import EnergyData
from .polynomial import harmonic_basis, harmonic_free_energy, free_energy, polynomial_basis


@dataclass(frozen=True)
class FitResult:
    """Result of the cubic free-energy fit."""

    theta: np.ndarray
    q_mean: np.ndarray
    q_std: np.ndarray
    q_scaled: np.ndarray
    design_matrix: np.ndarray
    delta_g_pbe: np.ndarray
    optimization_result: object


def _integration_grid(q_scaled: np.ndarray, params: Parameters):
    """Construct the 3D integration grid and its logarithmic weights."""
    q_min = np.min(q_scaled, axis=0)
    q_max = np.max(q_scaled, axis=0)
    padding = params.fit_padding_factor * (q_max - q_min)
    box_min = q_min - padding
    box_max = q_max + padding

    q1_grid = np.linspace(box_min[0], box_max[0], params.n_grid)
    q2_grid = np.linspace(box_min[1], box_max[1], params.n_grid)
    q3_grid = np.linspace(box_min[2], box_max[2], params.n_grid)
    dq1 = q1_grid[1] - q1_grid[0]
    dq2 = q2_grid[1] - q2_grid[0]
    dq3 = q3_grid[1] - q3_grid[0]

    w1 = np.ones(params.n_grid) * dq1
    w2 = np.ones(params.n_grid) * dq2
    w3 = np.ones(params.n_grid) * dq3
    w1[[0, -1]] *= 0.5
    w2[[0, -1]] *= 0.5
    w3[[0, -1]] *= 0.5

    Q1, Q2, Q3 = np.meshgrid(
        q1_grid, q2_grid, q3_grid, indexing="ij"
    )
    q_integration = np.column_stack(
        [Q1.ravel(), Q2.ravel(), Q3.ravel()]
    )
    if params.model == "squared_cubic":
        X_integration = polynomial_basis(q_integration)
    elif params.model == "harmonic":
        X_integration = harmonic_basis(q_integration)
    else:
        raise ValueError(f"Unknown free-energy model: {params.model}")

    W1, W2, W3 = np.meshgrid(w1, w2, w3, indexing="ij")
    integration_weights = (W1 * W2 * W3).ravel()
    log_integration_weights = np.log(integration_weights)

    return X_integration, log_integration_weights


def fit_free_energy(
    data: EnergyData,
    params: Parameters,
    initial_theta: np.ndarray | None = None,
) -> FitResult:
    """Fit the selected free-energy model."""
    q = data.q
    q_mean = np.mean(q, axis=0)
    q_std = np.std(q, axis=0)
    q_scaled = (q - q_mean) / q_std

    if params.model == "squared_cubic":
        X = polynomial_basis(q_scaled)
        evaluate_free_energy = free_energy
        n_parameters = 20
    elif params.model == "harmonic":
        X = harmonic_basis(q_scaled)
        evaluate_free_energy = harmonic_free_energy
        n_parameters = 10
    else:
        raise ValueError(f"Unknown free-energy model: {params.model}")

    X_integration, log_integration_weights = _integration_grid(
        q_scaled, params
    )

    def negative_log_likelihood(theta):
        """Evaluate the dimensionless negative log-likelihood for ``theta``."""
        # Evaluate the fitted free energy at the sampled configurations and
        # at every point on the integration grid used to normalize the model.
        G_data = evaluate_free_energy(theta, X)
        G_grid = evaluate_free_energy(theta, X_integration)
        if not np.all(np.isfinite(G_data)) or not np.all(np.isfinite(G_grid)):
            return np.inf

        # Compute log(Z), where Z = integral exp(-beta * G(q)) dq.
        # logsumexp keeps this normalization numerically stable when the
        # Boltzmann factors become very small.
        log_Z = logsumexp(
            -params.beta * G_grid + log_integration_weights
        )
        if not np.isfinite(log_Z):
            return np.inf

        # Negative log-likelihood:
        #   beta * sum_i G(q_i) + N * log(Z)
        # The returned value is dimensionless; it is not a free energy in Ha.
        return params.beta * np.sum(G_data) + len(q_scaled) * log_Z

    if initial_theta is None:
        theta0 = np.zeros(n_parameters)
        theta0[0] = np.sqrt(params.kBT)
    else:
        theta0 = np.asarray(initial_theta, dtype=float)
        if theta0.shape != (n_parameters,):
            raise ValueError(
                f"initial_theta must have shape ({n_parameters},), "
                f"got {theta0.shape}"
            )

    print("Starting optimization...")
    print(f"Number of samples: {len(q)}")
    print(f"Temperature:       {params.temperature:.2f} K")
    print(f"kBT:               {params.kBT:.8e} Ha")
    print(f"beta:              {params.beta:.8e} Ha^-1")
    print(f"Free-energy model: {params.model}")

    result = minimize(
        negative_log_likelihood,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-8, "maxls": 50},
    )
    print(result)
    if not result.success:
        print("WARNING: optimization did not fully converge.")

    theta = result.x
    delta_g_pbe = evaluate_free_energy(theta, X)

    return FitResult(
        theta=theta,
        q_mean=q_mean,
        q_std=q_std,
        q_scaled=q_scaled,
        design_matrix=X,
        delta_g_pbe=delta_g_pbe,
        optimization_result=result,
    )