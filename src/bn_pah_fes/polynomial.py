import numpy as np


def polynomial_basis(q: np.ndarray) -> np.ndarray:
    """Return the 20-term cubic polynomial basis."""
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    return np.column_stack([
        np.ones(len(q)), x, y, z,
        x**2, y**2, z**2, x * y, x * z, y * z,
        x**3, y**3, z**3,
        x**2 * y, x**2 * z, y**2 * x, y**2 * z,
        z**2 * x, z**2 * y, x * y * z,
    ])


def polynomial(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Evaluate the polynomial for a design matrix X."""
    return X @ theta


def free_energy(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Evaluate the squared polynomial free-energy model."""
    return polynomial(theta, X) ** 2
