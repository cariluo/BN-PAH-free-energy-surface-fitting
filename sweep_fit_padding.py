from dataclasses import replace
from pathlib import Path

import numpy as np

from bn_pah_fes.config import Parameters
from bn_pah_fes.data import load_data
from bn_pah_fes.fitting import fit_free_energy
from bn_pah_fes.kde import calculate_kde_surface
from bn_pah_fes.plotting import (
    plot_3d_free_energy_surface,
    plot_coordinate_time_series,
    plot_delta_g_contour,
    plot_kde_surface,
)
from bn_pah_fes.surfaces import calculate_surfaces, find_surface_minima


FIT_PADDING_FACTORS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
RESULTS_DIR = Path("results")


params = Parameters(model="harmonic")
data = load_data(Path("data"), params.n_samples, params.n_acf)
q = data.q
samples_in_fit = data.samples_in_fit

for fit_padding_factor in FIT_PADDING_FACTORS:
    params_run = replace(params, fit_padding_factor=fit_padding_factor)
    run_dir = RESULTS_DIR / f"fit_padding_{fit_padding_factor:g}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nfit_padding_factor = {fit_padding_factor:g}")
    print(f"Using {samples_in_fit} samples in the fitting")

    # Plot coordinate trajectories.
    plot_coordinate_time_series(data, run_dir)

    # Weighted empirical 2D free-energy surface.
    kde_result = calculate_kde_surface(data, params_run)
    print("G range:", kde_result.free_energy.min(), kde_result.free_energy.max())
    print(
        "G_sampled range:",
        kde_result.sampled_free_energy.min(),
        kde_result.sampled_free_energy.max(),
    )
    plot_kde_surface(data, kde_result, run_dir)

    # Free-energy fit.
    fit_result = fit_free_energy(data, params_run)
    delta_g_pbe = fit_result.delta_g_pbe

    output_data = np.column_stack(
        [data.idx, q[:, 0], q[:, 1], q[:, 2], delta_g_pbe]
    )
    np.savetxt(
        run_dir
        / (
            f"DeltaG_PBE_n_grid_{params_run.n_grid}_"
            f"fit_padding_factor_{fit_padding_factor:g}.txt"
        ),
        output_data,
        fmt=["%d", "%.10e", "%.10e", "%.10e", "%.10e"],
        header="index q0 qS qT DeltaG_PBE_Hartree",
    )

    # Reweighted 2D surfaces.
    surface_result = calculate_surfaces(fit_result, q, params_run)

    # Find continuous minima.
    minima = find_surface_minima(
        fit_result, q, params_run, surface_result
    )
    for surface, minimum in minima.items():
        print(
            f"{surface} minimum: "
            f"qS = {minimum.qS:.8f} Ha, "
            f"qT = {minimum.qT:.8f} Ha, "
            f"G = {minimum.value:.8f} Ha"
        )

    # Plot surfaces.
    plot_delta_g_contour(
        surface_result.QS,
        surface_result.QT,
        surface_result.G0_surface,
        q,
        "DeltaG0_contour.png",
        r"$\\Delta G_0$ (Ha)",
        run_dir,
        samples_in_fit,
    )
    plot_delta_g_contour(
        surface_result.QS,
        surface_result.QT,
        surface_result.GS_surface,
        q,
        "DeltaGS_contour.png",
        r"$\\Delta G_S$ (Ha)",
        run_dir,
        samples_in_fit,
    )
    plot_delta_g_contour(
        surface_result.QS,
        surface_result.QT,
        surface_result.GT_surface,
        q,
        "DeltaGT_contour.png",
        r"$\\Delta G_T$ (Ha)",
        run_dir,
        samples_in_fit,
    )
    plot_3d_free_energy_surface(
        surface_result.QS,
        surface_result.QT,
        surface_result.G0_surface,
        q,
        surface_result.G0_sampled,
        "DeltaG0_surface.png",
        r"$\\Delta G_0(q_S,q_T)$ (Ha)",
        run_dir,
        samples_in_fit,
    )
    plot_3d_free_energy_surface(
        surface_result.QS,
        surface_result.QT,
        surface_result.GS_surface,
        q,
        surface_result.GS_sampled,
        "DeltaGS_surface.png",
        r"$\\Delta G_S(q_S,q_T)$ (Ha)",
        run_dir,
        samples_in_fit,
    )
    plot_3d_free_energy_surface(
        surface_result.QS,
        surface_result.QT,
        surface_result.GT_surface,
        q,
        surface_result.GT_sampled,
        "DeltaGT_surface.png",
        run_dir,
        samples_in_fit,
    )
