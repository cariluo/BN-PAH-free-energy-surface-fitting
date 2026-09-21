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
    plot_energy_correlation,
)
from bn_pah_fes.surfaces import calculate_surfaces, find_surface_minima


# Parameters
params = Parameters(model="harmonic")
N_SAMPLES = params.n_samples
N_ACF = params.n_acf
fit_padding_factor = params.fit_padding_factor
n_grid = params.n_grid

# Output directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Read energies
data = load_data(Path("data"), N_SAMPLES, N_ACF)
q = data.q
samples_in_fit = data.samples_in_fit
print(f"Using {samples_in_fit} samples in the fitting")

# Plot coordinate trajectories
plot_coordinate_time_series(data, RESULTS_DIR)

# Plot correlation between PBE0 energy and S0 energy
plt.figure(figsize=(8, 6))
plt.scatter(
    data.E_pbe,
    data.S0,
    s=20,
    facecolors="white",
    edgecolors="black",
    alpha=0.8,
)

slope, intercept = np.polyfit(data.E_pbe, data.S0, 1)
x_fit = np.linspace(data.E_pbe.min(), data.E_pbe.max(), 200)
plt.plot(
    x_fit,
    slope * x_fit + intercept,
    linewidth=2,
    label=fr"$r = {np.corrcoef(data.E_pbe, data.S0)[0, 1]:.4f}$",
)

plt.xlabel(r"PBE0 energy (Ha)", fontsize=30)
plt.ylabel(r"$S_0$ energy (Ha)", fontsize=30)
plt.tick_params(axis="both", labelsize=18)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "PBE0_S0_correlation.png", dpi=300, bbox_inches="tight")
plt.close()

# Weighted empirical 2D free-energy surface
kde_result = calculate_kde_surface(data, params)
print("G range:", kde_result.free_energy.min(), kde_result.free_energy.max())
print(
    "G_sampled range:",
    kde_result.sampled_free_energy.min(),
    kde_result.sampled_free_energy.max(),
)
plot_kde_surface(data, kde_result, RESULTS_DIR)

# Cubic polynomial fit
fit_result = fit_free_energy(data, params)
DeltaG_PBE = fit_result.delta_g_pbe

output_data = np.column_stack([data.idx, q[:, 0], q[:, 1], q[:, 2], DeltaG_PBE])
np.savetxt(
    RESULTS_DIR / f"DeltaG_PBE_n_grid_{n_grid}_fit_padding_factor_{fit_padding_factor}.txt",
    output_data,
    fmt=["%d", "%.10e", "%.10e", "%.10e", "%.10e"],
    header="index q0 qS qT DeltaG_PBE_Hartree",
)

# Reweighted 2D surfaces
surface_result = calculate_surfaces(fit_result, q, params)
QS = surface_result.QS
QT = surface_result.QT
G0_surface = surface_result.G0_surface
GS_surface = surface_result.GS_surface
GT_surface = surface_result.GT_surface

# Surface values evaluated at the subsampled trajectory points.
# These are used to overlay the sampled configurations on the 2D/3D surfaces.
G0_sampled = surface_result.G0_sampled
GS_sampled = surface_result.GS_sampled
GT_sampled = surface_result.GT_sampled

# Find continuous minima of the fitted surfaces.
minima = find_surface_minima(fit_result, q, params, surface_result)
for surface, minimum in minima.items():
    print(
        f"{surface} minimum: "
        f"qS = {minimum.qS:.8f} Ha, "
        f"qT = {minimum.qT:.8f} Ha, "
        f"G = {minimum.value:.8f} Ha"
    )

# Plot surfaces
plot_delta_g_contour(
    QS,
    QT,
    G0_surface,
    q,
    "DeltaG0_contour.png",
    r"$\Delta G_0$ (Ha)",
    RESULTS_DIR,
    samples_in_fit,
    minimum=(minima["G0"].qS, minima["G0"].qT, minima["G0"].value),
)
plot_delta_g_contour(
    QS,
    QT,
    GS_surface,
    q,
    "DeltaGS_contour.png",
    r"$\Delta G_S$ (Ha)",
    RESULTS_DIR,
    samples_in_fit,
    minimum=(minima["GS"].qS, minima["GS"].qT, minima["GS"].value),
)
plot_delta_g_contour(
    QS,
    QT,
    GT_surface,
    q,
    "DeltaGT_contour.png",
    r"$\Delta G_T$ (Ha)",
    RESULTS_DIR,
    samples_in_fit,
    minimum=(minima["GT"].qS, minima["GT"].qT, minima["GT"].value),
)
plot_3d_free_energy_surface(
    QS,
    QT,
    G0_surface,
    q,
    G0_sampled,
    "DeltaG0_surface.png",
    r"$\Delta G_0(q_S,q_T)$ (Ha)",
    RESULTS_DIR,
    samples_in_fit,
    minimum=(minima["G0"].qS, minima["G0"].qT, minima["G0"].value),
)
plot_3d_free_energy_surface(
    QS,
    QT,
    GS_surface,
    q,
    GS_sampled,
    "DeltaGS_surface.png",
    r"$\Delta G_S(q_S,q_T)$ (Ha)",
    RESULTS_DIR,
    samples_in_fit,
    minimum=(minima["GS"].qS, minima["GS"].qT, minima["GS"].value),
)
plot_3d_free_energy_surface(
    QS,
    QT,
    GT_surface,
    q,
    GT_sampled,
    "DeltaGT_surface.png",
    r"$\Delta G_T(q_S,q_T)$ (Ha)",
    RESULTS_DIR,
    samples_in_fit,
    minimum=(minima["GT"].qS, minima["GT"].qT, minima["GT"].value),
)
