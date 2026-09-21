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
    plot_energy_correlation,
    plot_kde_surface,
    plot_singlet_triplet_gap_histogram,
)
from bn_pah_fes.polynomial import harmonic_free_energy, polynomial_basis
from bn_pah_fes.surfaces import calculate_surfaces, find_surface_minima


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    # Fit the harmonic model first.
    harmonic_params = Parameters(model="harmonic")
    data = load_data(
        DATA_DIR,
        harmonic_params.n_samples,
        harmonic_params.n_acf,
    )
    harmonic_result = fit_free_energy(data, harmonic_params)

    # Construct a squared-cubic initial guess from the harmonic result.
    q_scaled = harmonic_result.q_scaled
    X_cubic = polynomial_basis(q_scaled)
    harmonic_G = harmonic_free_energy(
        harmonic_result.theta,
        harmonic_result.design_matrix,
    )
    target = np.sqrt(np.maximum(harmonic_G, 0.0))
    cubic_initial_theta, *_ = np.linalg.lstsq(
        X_cubic,
        target,
        rcond=None,
    )

    # Fit the squared-cubic model from the harmonic initial guess.
    params = Parameters(model="squared_cubic")
    fit_result = fit_free_energy(
        data,
        params,
        initial_theta=cubic_initial_theta,
    )

    q = data.q
    samples_in_fit = data.samples_in_fit
    fit_padding_factor = params.fit_padding_factor
    n_grid = params.n_grid

    # Plot the same results as fit_free_energy.py.
    plot_coordinate_time_series(data, RESULTS_DIR)
    plot_energy_correlation(data, RESULTS_DIR)
    plot_singlet_triplet_gap_histogram(data, RESULTS_DIR)

    kde_result = calculate_kde_surface(data, params)
    plot_kde_surface(data, kde_result, RESULTS_DIR)

    DeltaG_PBE = fit_result.delta_g_pbe
    output_data = np.column_stack(
        [data.idx, q[:, 0], q[:, 1], q[:, 2], DeltaG_PBE]
    )
    np.savetxt(
        RESULTS_DIR
        / f"DeltaG_PBE_n_grid_{n_grid}_fit_padding_factor_{fit_padding_factor}.txt",
        output_data,
        fmt=["%d", "%.10e", "%.10e", "%.10e", "%.10e"],
        header="index q0 qS qT DeltaG_PBE_Hartree",
    )

    # Calculate and plot the reweighted free-energy surfaces.
    surface_result = calculate_surfaces(fit_result, q, params)
    QS = surface_result.QS
    QT = surface_result.QT
    G0_surface = surface_result.G0_surface
    GS_surface = surface_result.GS_surface
    GT_surface = surface_result.GT_surface

    G0_sampled = surface_result.G0_sampled
    GS_sampled = surface_result.GS_sampled
    GT_sampled = surface_result.GT_sampled

    minima = find_surface_minima(
        fit_result,
        q,
        params,
        surface_result,
    )
    for surface, minimum in minima.items():
        print(
            f"{surface} minimum: "
            f"qS = {minimum.qS:.8f} Ha, "
            f"qT = {minimum.qT:.8f} Ha, "
            f"G = {minimum.value:.8f} Ha"
        )

    plot_delta_g_contour(
        QS, QT, G0_surface, q, "DeltaG0_contour.png",
        r"$\Delta G_0$ (Ha)", RESULTS_DIR, samples_in_fit,
        minimum=(minima["G0"].qS, minima["G0"].qT, minima["G0"].value),
    )
    plot_delta_g_contour(
        QS, QT, GS_surface, q, "DeltaGS_contour.png",
        r"$\Delta G_S$ (Ha)", RESULTS_DIR, samples_in_fit,
        minimum=(minima["GS"].qS, minima["GS"].qT, minima["GS"].value),
    )
    plot_delta_g_contour(
        QS, QT, GT_surface, q, "DeltaGT_contour.png",
        r"$\Delta G_T$ (Ha)", RESULTS_DIR, samples_in_fit,
        minimum=(minima["GT"].qS, minima["GT"].qT, minima["GT"].value),
    )
    plot_3d_free_energy_surface(
        QS, QT, G0_surface, q, G0_sampled, "DeltaG0_surface.png",
        r"$\Delta G_0(q_S,q_T)$ (Ha)", RESULTS_DIR, samples_in_fit,
        minimum=(minima["G0"].qS, minima["G0"].qT, minima["G0"].value),
    )
    plot_3d_free_energy_surface(
        QS, QT, GS_surface, q, GS_sampled, "DeltaGS_surface.png",
        r"$\Delta G_S(q_S,q_T)$ (Ha)", RESULTS_DIR, samples_in_fit,
        minimum=(minima["GS"].qS, minima["GS"].qT, minima["GS"].value),
    )
    plot_3d_free_energy_surface(
        QS, QT, GT_surface, q, GT_sampled, "DeltaGT_surface.png",
        r"$\Delta G_T(q_S,q_T)$ (Ha)", RESULTS_DIR, samples_in_fit,
        minimum=(minima["GT"].qS, minima["GT"].qT, minima["GT"].value),
    )


if __name__ == "__main__":
    main()
