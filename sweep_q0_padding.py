import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from bn_pah_fes.config import Parameters
from bn_pah_fes.data import load_data
from bn_pah_fes.fitting import fit_free_energy
from bn_pah_fes.surfaces import calculate_surfaces


Q0_PADDING_FACTORS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
RESULTS_DIR = Path("results")
SUMMARY_FILE = RESULTS_DIR / "q0_padding_summary.csv"


def main() -> None:
    """Test convergence of G0 with respect to q0 integration padding."""
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

    surfaces = []
    for q0_padding_factor in Q0_PADDING_FACTORS:
        surface_params = replace(
            params,
            q0_padding_factor=q0_padding_factor,
        )
        print(f"Calculating G0 for q0_padding_factor = {q0_padding_factor:g}")
        surface_result = calculate_surfaces(
            fit_result,
            data.q,
            surface_params,
        )
        surfaces.append(surface_result.G0_surface)

    reference_surface = surfaces[-1]
    rows = []

    previous_surface = None
    for q0_padding_factor, surface in zip(Q0_PADDING_FACTORS, surfaces):
        max_change_from_previous = np.nan
        if previous_surface is not None:
            max_change_from_previous = float(
                np.max(np.abs(surface - previous_surface))
            )

        max_change_from_reference = float(
            np.max(np.abs(surface - reference_surface))
        )

        rows.append(
            {
                "q0_padding_factor": q0_padding_factor,
                "max_change_from_previous_Ha": max_change_from_previous,
                "max_change_from_reference_Ha": max_change_from_reference,
            }
        )
        previous_surface = surface

    with SUMMARY_FILE.open("w", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "q0_padding_factor",
                "max_change_from_previous_Ha",
                "max_change_from_reference_Ha",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary written to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
