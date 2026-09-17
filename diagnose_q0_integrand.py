from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bn_pah_fes.config import Parameters
from bn_pah_fes.data import load_data
from bn_pah_fes.fitting import fit_free_energy
from bn_pah_fes.surfaces import calculate_q0_integrand, calculate_surfaces


Q0_PADDING_FACTORS = [0.1, 0.5, 1.0, 2.0]
RESULTS_DIR = Path("results")


def main() -> None:
    """Plot the normalized q0 integrand at representative surface points."""
    base_params = Parameters()
    params = replace(base_params, fit_padding_factor=0.0)

    data = load_data(
        Path("data"),
        params.n_samples,
        params.n_acf,
    )

    print("Fitting MLE with fit_padding_factor = 0.0")
    fit_result = fit_free_energy(data, params)

    # Use the surface grid to define the G0 minimum and four corners.
    surface_result = calculate_surfaces(
        fit_result,
        data.q,
        params,
    )

    minimum_index = np.unravel_index(
        np.argmin(surface_result.G0_surface),
        surface_result.G0_surface.shape,
    )
    minimum_point = (
        surface_result.QS[minimum_index],
        surface_result.QT[minimum_index],
    )

    qS_min = surface_result.qS_grid.min()
    qS_max = surface_result.qS_grid.max()
    qT_min = surface_result.qT_grid.min()
    qT_max = surface_result.qT_grid.max()

    points = {
        "G0_min": minimum_point,
        "lower_left": (qS_min, qT_min),
        "lower_right": (qS_min, qT_max),
        "upper_left": (qS_max, qT_min),
        "upper_right": (qS_max, qT_max),
    }

    RESULTS_DIR.mkdir(exist_ok=True)

    for name, (qS_value, qT_value) in points.items():
        print(
            f"\nPoint: {name}, "
            f"qS = {qS_value:.6f}, qT = {qT_value:.6f}"
        )

        fig, ax = plt.subplots()

        for padding in Q0_PADDING_FACTORS:
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
                params,
            )

            ax.plot(
                q0_grid,
                integrand,
                label=f"padding = {padding:g}",
            )

            print(
                f"  padding = {padding:g}: "
                f"left boundary = {integrand[0]:.3e}, "
                f"right boundary = {integrand[-1]:.3e}"
            )

        ax.set_xlabel(r"$q_0$")
        ax.set_ylabel(r"Normalized integrand")
        ax.set_yscale("log")
        ax.set_title(
            rf"$q_S={qS_value:.3f},\ q_T={qT_value:.3f}$ ({name})"
        )
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            RESULTS_DIR / f"q0_integrand_{name}.png",
            dpi=300,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
