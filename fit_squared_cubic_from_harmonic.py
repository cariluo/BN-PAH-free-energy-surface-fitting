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
    plot_optimizer_performance,
    plot_singlet_triplet_gap_histogram,
)
from bn_pah_fes.polynomial import harmonic_free_energy, polynomial_basis
from bn_pah_fes.surfaces import calculate_surfaces, find_surface_minima

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def main() -> None:
    harmonic_params = Parameters(model="harmonic")
    data = load_data(DATA_DIR, harmonic_params.n_samples, harmonic_params.n_acf)
    harmonic_result = fit_free_energy(data, harmonic_params)

    q_scaled = harmonic_result.q_scaled
    X_cubic = polynomial_basis(q_scaled)
    harmonic_G = harmonic_free_energy(harmonic_result.theta, harmonic_result.design_matrix)
    target = np.sqrt(np.maximum(harmonic_G, 0.0))
    cubic_initial_theta, *_ = np.linalg.lstsq(X_cubic, target, rcond=None)

    params = Parameters(model="squared_cubic")
    fit_result = fit_free_energy(data, params, initial_theta=cubic_initial_theta)

    q = data.q
    samples_in_fit = data.samples_in_fit
    plot_coordinate_time_series(data, RESULTS_DIR)
    plot_energy_correlation(data, RESULTS_DIR)
    plot_singlet_triplet_gap_histogram(data, RESULTS_DIR)
    plot_optimizer_performance(fit_result.optimization_history, RESULTS_DIR)

    kde_result = calculate_kde_surface(data, params)
    plot_kde_surface(data, kde_result, RESULTS_DIR)

    DeltaG_PBE = fit_result.delta_g_pbe
    np.savetxt(
        RESULTS_DIR / f"DeltaG_PBE_n_grid_{params.n_grid}_fit_padding_factor_{params.fit_padding_factor}.txt",
        np.column_stack([data.idx, q[:, 0], q[:, 1], q[:, 2], DeltaG_PBE]),
        fmt=["%d", "%.10e", "%.10e", "%.10e", "%.10e"],
        header="index q0 qS qT DeltaG_PBE_Hartree",
    )

    surface_result = calculate_surfaces(fit_result, q, params)
    minima = find_surface_minima(fit_result, q, params, surface_result)
    for surface, minimum in minima.items():
        print(f"{surface} minimum: qS = {minimum.qS:.8f} Ha, qT = {minimum.qT:.8f} Ha, G = {minimum.value:.8f} Ha")

    plot_data = [
        ("G0", surface_result.G0_surface, surface_result.G0_sampled, "DeltaG0_contour.png", "DeltaG0_surface.png", r"$\Delta G_0(q_S,q_T)$ (Ha)"),
        ("GS", surface_result.GS_surface, surface_result.GS_sampled, "DeltaGS_contour.png", "DeltaGS_surface.png", r"$\Delta G_S(q_S,q_T)$ (Ha)"),
        ("GT", surface_result.GT_surface, surface_result.GT_sampled, "DeltaGT_contour.png", "DeltaGT_surface.png", r"$\Delta G_T(q_S,q_T)$ (Ha)"),
    ]
    for key, surface, sampled, contour_file, surface_file, zlabel in plot_data:
        plot_delta_g_contour(
            surface_result.QS, surface_result.QT, surface, q,
            contour_file, zlabel, RESULTS_DIR, samples_in_fit,
            minimum=(minima[key].qS, minima[key].qT, minima[key].value),
        )
        plot_3d_free_energy_surface(
            surface_result.QS, surface_result.QT, surface, q, sampled,
            surface_file, zlabel, RESULTS_DIR, samples_in_fit,
            minimum=(minima[key].qS, minima[key].qT, minima[key].value),
        )

if __name__ == "__main__":
    main()
