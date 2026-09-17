from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bn_pah_fes.config import Parameters
from bn_pah_fes.data import load_data
from bn_pah_fes.fitting import fit_free_energy
from bn_pah_fes.surfaces import calculate_q0_integrand


Q0_PADDING_FACTORS = [0.0, 0.1, 0.5, 1.0, 2.0]


def main() -> None:
    """Plot the q0 integrand for several q0 integration paddings."""
    base_params = Parameters()
    params = replace(base_params, fit_padding_factor=0.0)

    data = load_data(
        Path("data"),
        params.n_samples,
        params.n_acf,
    )

    print("Fitting MLE with fit_padding_factor = 0.0")
    fit_result = fit_free_energy(data, params)

    # Use the G0 minimum as one representative (qS, qT) point.
    surface_result = calculate_surfaces(
        fit_result,
        data.q,
        params,
    )

    qS_value = surface_result.QS.ravel()[
        np.argmin(surface_result.G0_surface)
    ]
    qT_value = surface_result.QT.ravel()[
        np.argmin(surface_result.G0_surface)
    ]

    print(f"Using qS = {qS_value:.6f}, qT = {qT_value:.6f}")

    fig, ax = plt.subplots()

    for padding in Q0_PADDING_FACTORS:
        diagnostic_params = replace(
            params,
            q0_padding_factor=padding,
        )

        q0_min = data.q[:, 0].min()
        q0_max = data.q[:, 0].max()
        q0_pad = padding * (q0_max - q0_min)

        q0_grid = np.linspace(
            q0_min - q0_pad,
            q0_max + q0_pad,
            params.n_q0,
        )

        integrand = calculate_q0_integrand(
            qS_value,
            qT_value,
            q0_grid,
            fit_result,
            diagnostic_params,
        )

        ax.plot(
            q0_grid,
            integrand,
            label=f"padding = {padding:g}",
        )

        print(
            f"padding = {padding:g}: "
            f"left boundary = {integrand[0]:.3e}, "
            f"right boundary = {integrand[-1]:.3e}"
        )

    ax.set_xlabel(r"$q_0$")
    ax.set_ylabel(r"Normalized integrand")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig("q0_integrand_diagnostic.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
