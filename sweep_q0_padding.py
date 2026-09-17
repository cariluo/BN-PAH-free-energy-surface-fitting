from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

from bn_pah_fes.config import Parameters
from bn_pah_fes.data import load_data
from bn_pah_fes.fitting import fit_free_energy
from bn_pah_fes.plotting import (
    plot_3d_free_energy_surface,
    plot_delta_g_contour,
)
from bn_pah_fes.surfaces import calculate_surfaces, find_surface_minima


Q0_PADDING_FACTORS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
RESULTS_DIR = Path("results")


def main() -> None:
    """Save G0 plots for a series of q0 integration paddings."""
    base_params = Parameters()
    params = replace(base_params, fit_padding_factor=0.0)

    data = load_data(
        Path("data"),
        params.n_samples,
        params.n_acf,
    )
    RESULTS_DIR.mkdir(exist_ok=True)

    # Fit the MLE model once. q0_padding_factor only affects the subsequent
    # numerical integration over q0, so refitting is unnecessary.
    print("Fitting MLE with fit_padding_factor = 0.0")
    fit_result = fit_free_energy(data, params)

    previous_surface = None
    previous_minimum = None
    convergence_results = []

    for q0_padding_factor in Q0_PADDING_FACTORS:
        surface_params = replace(
            params,
            q0_padding_factor=q0_padding_factor,
        )
        run_dir = RESULTS_DIR / f"q0_padding_{q0_padding_factor:g}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nq0_padding_factor = {q0_padding_factor:g}")
        print(f"Results directory: {run_dir}")

        surface_result = calculate_surfaces(
            fit_result,
            data.q,
            surface_params,
        )

        # Find the minimum of the current G0 surface.
        minima = find_surface_minima(surface_result)

        current_minimum = np.array([
            minima["G0"][0],
            minima["G0"][1],
        ])

        # Compare against the previous q0-padding value.
        if previous_surface is None:
            max_abs_change = np.nan
            rms_change = np.nan
            minimum_location_change = np.nan
        else:
            surface_difference = (
                surface_result.G0_surface - previous_surface
            )

            # Metric 1: maximum absolute change in G0.
            max_abs_change = np.max(np.abs(surface_difference))

            # Metric 2: RMS change in G0.
            rms_change = np.sqrt(
                np.mean(surface_difference**2)
            )

            # Metric 3: Euclidean distance between surface minima.
            minimum_location_change = np.linalg.norm(
                current_minimum - previous_minimum
            )

        convergence_results.append({
            "q0_padding_factor": q0_padding_factor,
            "max_abs_change_Ha": max_abs_change,
            "rms_change_Ha": rms_change,
            "minimum_location_change": minimum_location_change,
            "minimum_qS": current_minimum[0],
            "minimum_qT": current_minimum[1],
        })

        previous_surface = surface_result.G0_surface.copy()
        previous_minimum = current_minimum.copy()

        plot_delta_g_contour(
            surface_result.QS,
            surface_result.QT,
            surface_result.G0_surface,
            data.q,
            "DeltaG0_contour.png",
            r"$\Delta G_0$ (Ha)",
            run_dir,
            data.samples_in_fit,
        )

        plot_3d_free_energy_surface(
            surface_result.QS,
            surface_result.QT,
            surface_result.G0_surface,
            data.q,
            surface_result.G0_sampled,
            "DeltaG0_surface.png",
            r"$\Delta G_0(q_S,q_T)$ (Ha)",
            run_dir,
            data.samples_in_fit,
        )

    print("\nq0-padding plots complete.")
    
    convergence_df = pd.DataFrame(convergence_results)

    convergence_df.to_csv(
        RESULTS_DIR / "q0_padding_convergence.csv",
        index=False,
    )

    print(
        "\nSaved convergence metrics to "
        f"{RESULTS_DIR / 'q0_padding_convergence.csv'}"
    )

if __name__ == "__main__":
    main()
