import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from bn_pah_fes.config import Parameters
from bn_pah_fes.data import load_data
from bn_pah_fes.fitting import fit_free_energy
from bn_pah_fes.plotting import (
    plot_3d_free_energy_surface,
    plot_delta_g_contour,
)
from bn_pah_fes.surfaces import calculate_surfaces, find_surface_minima


FIT_PADDING_FACTORS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
RESULTS_DIR = Path("results")


def main() -> None:
    """Run the free-energy fit for a series of integration-grid paddings."""
    base_params = Parameters()
    data = load_data(
        Path("data"),
        base_params.n_samples,
        base_params.n_acf,
    )
    RESULTS_DIR.mkdir(exist_ok=True)

    summary_rows = []

    for fit_padding_factor in FIT_PADDING_FACTORS:
        params = replace(
            base_params,
            fit_padding_factor=fit_padding_factor,
        )

        run_dir = RESULTS_DIR / f"fit_padding_{fit_padding_factor:g}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nfit_padding_factor = {fit_padding_factor:g}")
        print(f"Results directory: {run_dir}")

        fit_result = fit_free_energy(data, params)
        surface_result = calculate_surfaces(fit_result, data.q, params)
        minima = find_surface_minima(
            fit_result,
            data.q,
            params,
            surface_result=surface_result,
        )

        output_data = np.column_stack(
            [
                data.idx,
                data.q[:, 0],
                data.q[:, 1],
                data.q[:, 2],
                fit_result.delta_g_pbe,
            ]
        )
        np.savetxt(
            run_dir
            / (
                "DeltaG_PBE_"
                f"n_grid_{params.n_grid}_"
                f"fit_padding_factor_{fit_padding_factor:g}.txt"
            ),
            output_data,
            fmt=["%d", "%.10e", "%.10e", "%.10e", "%.10e"],
            header="index q0 qS qT DeltaG_PBE_Hartree",
        )

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
        plot_delta_g_contour(
            surface_result.QS,
            surface_result.QT,
            surface_result.GS_surface,
            data.q,
            "DeltaGS_contour.png",
            r"$\Delta G_S$ (Ha)",
            run_dir,
            data.samples_in_fit,
        )
        plot_delta_g_contour(
            surface_result.QS,
            surface_result.QT,
            surface_result.GT_surface,
            data.q,
            "DeltaGT_contour.png",
            r"$\Delta G_T$ (Ha)",
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
        plot_3d_free_energy_surface(
            surface_result.QS,
            surface_result.QT,
            surface_result.GS_surface,
            data.q,
            surface_result.GS_sampled,
            "DeltaGS_surface.png",
            r"$\Delta G_S(q_S,q_T)$ (Ha)",
            run_dir,
            data.samples_in_fit,
        )
        plot_3d_free_energy_surface(
            surface_result.QS,
            surface_result.QT,
            surface_result.GT_surface,
            data.q,
            surface_result.GT_sampled,
            "DeltaGT_surface.png",
            r"$\Delta G_T(q_S,q_T)$ (Ha)",
            run_dir,
            data.samples_in_fit,
        )

        row = {"fit_padding_factor": fit_padding_factor}
        for surface_name, minimum in minima.items():
            row[f"{surface_name}_qS"] = minimum.qS
            row[f"{surface_name}_qT"] = minimum.qT
            row[f"{surface_name}_min"] = minimum.value
            row[f"{surface_name}_success"] = minimum.success
        summary_rows.append(row)

    fieldnames = [
        "fit_padding_factor",
        "G0_qS",
        "G0_qT",
        "G0_min",
        "G0_success",
        "GS_qS",
        "GS_qT",
        "GS_min",
        "GS_success",
        "GT_qS",
        "GT_qT",
        "GT_min",
        "GT_success",
    ]
    with (RESULTS_DIR / "fit_padding_summary.csv").open(
        "w",
        newline="",
    ) as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary written to {RESULTS_DIR / 'fit_padding_summary.csv'}")


if __name__ == "__main__":
    main()
