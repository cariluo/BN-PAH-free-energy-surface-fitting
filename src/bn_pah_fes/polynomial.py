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


def polynomial_gradient(theta: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Evaluate the gradient of the cubic polynomial with respect to q."""
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    gradient_basis = np.column_stack([
        np.zeros(len(q)),
        np.ones(len(q)),
        np.zeros(len(q)),
        np.zeros(len(q)),
        2 * x, 0 * y, 0 * z, y, z, 0 * x,
        3 * x**2, 0 * y**2, 0 * z**2,
        2 * x * y, 2 * x * z, y**2, 0 * y**2, z**2, 0 * z**2, y * z,
    ]).reshape(len(q), 20)

    dx = np.column_stack([
        np.zeros(len(q)),
        np.ones(len(q)),
        np.zeros(len(q)),
        np.zeros(len(q)),
        2 * x, np.zeros(len(q)), np.zeros(len(q)),
        y, z, np.zeros(len(q)),
        3 * x**2, np.zeros(len(q)), np.zeros(len(q)),
        2 * x * y, 2 * x * z, y**2, np.zeros(len(q)), z**2,
        np.zeros(len(q)), y * z,
    ])
    dy = np.column_stack([
        np.zeros(len(q)),
        np.zeros(len(q)),
        np.ones(len(q)),
        np.zeros(len(q)),
        np.zeros(len(q)), 2 * y, np.zeros(len(q)),
        x, np.zeros(len(q)), z,
        np.zeros(len(q)), 3 * y**2, np.zeros(len(q)),
        x**2, np.zeros(len(q)), 2 * x * y, 2 * y * z,
        np.zeros(len(q)), z**2, x * z,
    ])
    dz = np.column_stack([
        np.zeros(len(q)),
        np.zeros(len(q)),
        np.zeros(len(q)),
        np.ones(len(q)),
        np.zeros(len(q)), np.zeros(len(q)), 2 * z,
        np.zeros(len(q)), x, y,
        np.zeros(len(q)), np.zeros(len(q)), 3 * z**2,
        np.zeros(len(q)), x**2, np.zeros(len(q)), y**2,
        2 * x * z, 2 * y * z, x * y,
    ])
    gradient_basis = np.stack([dx, dy, dz], axis=1)
    return np.einsum("ij,ijk->ik", theta, gradient_basis)


def free_energy(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Evaluate the squared polynomial free-energy model."""
    return polynomial(theta, X) ** 2
