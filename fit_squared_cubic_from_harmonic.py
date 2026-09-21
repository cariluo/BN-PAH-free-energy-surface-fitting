from pathlib import Path

import numpy as np

from bn_pah_fes.config import Parameters
from bn_pah_fes.data import load_data
from bn_pah_fes.fitting import fit_free_energy
from bn_pah_fes.polynomial import harmonic_free_energy, polynomial_basis


DATA_DIR = Path("data")


def main() -> None:
    # Fit the harmonic model first.
    harmonic_params = Parameters(model="harmonic")
    data = load_data(DATA_DIR, harmonic_params.n_samples, harmonic_params.n_acf)
    harmonic_result = fit_free_energy(data, harmonic_params)

    # Construct a squared-cubic initial guess by fitting its polynomial
    # inside the square to the square root of the harmonic free energy.
    q_scaled = harmonic_result.q_scaled
    X_cubic = polynomial_basis(q_scaled)
    harmonic_G = harmonic_free_energy(
        harmonic_result.theta,
        harmonic_result.design_matrix,
    )
    target = np.sqrt(np.maximum(harmonic_G, 0.0))
    cubic_initial_theta, *_ = np.linalg.lstsq(X_cubic, target, rcond=None)

    # Refine the squared-cubic model starting from that initial guess.
    cubic_params = Parameters(model="squared_cubic")
    cubic_result = fit_free_energy(
        data,
        cubic_params,
        initial_theta=cubic_initial_theta,
    )

    print("\nHarmonic fit:")
    print(harmonic_result.theta)
    print("\nSquared-cubic initial guess:")
    print(cubic_initial_theta)
    print("\nSquared-cubic fit:")
    print(cubic_result.theta)


if __name__ == "__main__":
    main()
