from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from bn_pah_fes.config import Parameters
from bn_pah_fes.data import load_data
from bn_pah_fes.fitting import fit_free_energy
from bn_pah_fes.kde import calculate_kde_surface
from bn_pah_fes.surfaces import calculate_surfaces

plt.rcParams["font.family"] = "Times New Roman"


# Parameters
params = Parameters()
N_SAMPLES = params.n_samples
N_ACF = params.n_acf
fit_padding_factor = params.fit_padding_factor
n_grid = params.n_grid

# Output directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Read energies
data = load_data(Path("data"), N_SAMPLES, N_ACF)
qS = data.qS
qT = data.qT
q0 = data.q0
qS_all = data.qS_all
qT_all = data.qT_all
q0_all = data.q0_all
idx = data.idx
idx_all = data.idx_all
q = data.q

samples_in_fit = data.samples_in_fit
print(f"Using {samples_in_fit} samples in the fitting")

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].scatter(idx_all, q0_all, s=2, label="All")
axes[0].scatter(idx, q0, s=16, label="Subsampled")
axes[0].set_ylabel(r"$q_0$", fontsize=30)
axes[0].legend(fontsize=10)
axes[1].scatter(idx_all, qS_all, s=2)
axes[1].scatter(idx, qS, s=16)
axes[1].set_ylabel(r"$q_S$", fontsize=30)
axes[2].scatter(idx_all, qT_all, s=2)
axes[2].scatter(idx, qT, s=16)
axes[2].set_ylabel(r"$q_T$", fontsize=30)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "coordinate_time_series.png", dpi=300)
plt.close()

# Weighted empirical 2D free-energy surface
kde_result = calculate_kde_surface(data, params)
QS = kde_result.QS
QT = kde_result.QT
G_empirical = kde_result.free_energy
G_sampled = kde_result.sampled_free_energy

print("G range:", G_empirical.min(), G_empirical.max())
print("G_sampled range:", G_sampled.min(), G_sampled.max())

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(QS, QT, G_empirical, cmap="viridis", edgecolor="none", alpha=0.7)
ax.scatter(qS, qT, G_sampled + 0.01, s=5, facecolor="white", edgecolor="black", alpha=0.5)
ax.set_xlabel(r"$q_S$", fontsize=30)
ax.set_ylabel(r"$q_T$", fontsize=30)
ax.set_zlabel(r"$-k_BT\ln P(q_S,q_T)$ (Ha)")
fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1, label=r"$-k_BT\ln P(q_S,q_T)$ (Ha)")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "KDE.png", dpi=300)
plt.close()

# Cubic polynomial fit
fit_result = fit_free_energy(data, params)
DeltaG_PBE = fit_result.delta_g_pbe

output_data = np.column_stack([idx, q[:, 0], q[:, 1], q[:, 2], DeltaG_PBE])
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
G0_sampled = surface_result.G0_sampled
GS_sampled = surface_result.GS_sampled
GT_sampled = surface_result.GT_sampled

# Plotting
def plot_3d_free_energy_surface(QS, QT, surface_data, q, sampled, filename, zlabel):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(QS, QT, surface_data, cmap="viridis", alpha=0.8, vmin=0.0, vmax=0.65)
    ax.scatter(q[:, 1], q[:, 2], sampled, facecolors="white", edgecolors="black", label=f"{samples_in_fit} sampled points", depthshade=False)
    ax.legend(loc="upper right")
    ax.set_xlabel(r"$q_S$ (Ha)", fontsize=20)
    ax.set_ylabel(r"$q_T$ (Ha)", fontsize=20)
    ax.set_zlabel(zlabel, fontsize=20)
    cbar = fig.colorbar(surface, ax=ax, shrink=0.7, pad=0.1, ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    cbar.set_label(zlabel, fontsize=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {RESULTS_DIR / filename}")


def plot_delta_g_contour(QS, QT, surface_data, q, filename, cbarlabel):
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(QS, QT, surface_data, levels=30, vmin=0.0, vmax=0.65, cmap="viridis")
    plt.scatter(q[:, 1], q[:, 2], facecolors="white", edgecolors="black", label=f"{samples_in_fit} sampled points")
    diag_min = max(QS.min(), QT.min())
    diag_max = min(QS.max(), QT.max())
    plt.plot([diag_min, diag_max], [diag_min, diag_max], color="white", linestyle="--", linewidth=2, label=r"$q_T=q_S$")
    plt.legend(loc="best")
    plt.xlabel(r"$q_S$ (Ha)", fontsize=20)
    plt.ylabel(r"$q_T$ (Ha)", fontsize=20)
    cbar = plt.colorbar(contour, ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    cbar.set_label(cbarlabel, fontsize=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {RESULTS_DIR / filename}")


plot_delta_g_contour(QS, QT, G0_surface, q, "DeltaG0_contour.png", r"$\Delta G_0$ (Ha)")
plot_delta_g_contour(QS, QT, GS_surface, q, "DeltaGS_contour.png", r"$\Delta G_S$ (Ha)")
plot_delta_g_contour(QS, QT, GT_surface, q, "DeltaGT_contour.png", r"$\Delta G_T$ (Ha)")
plot_3d_free_energy_surface(QS, QT, G0_surface, q, G0_sampled, "DeltaG0_surface.png", r"$\Delta G_0(q_S,q_T)$ (Ha)")
plot_3d_free_energy_surface(QS, QT, GS_surface, q, GS_sampled, "DeltaGS_surface.png", r"$\Delta G_S(q_S,q_T)$ (Ha)")
plot_3d_free_energy_surface(QS, QT, GT_surface, q, GT_sampled, "DeltaGT_surface.png", r"$\Delta G_T(q_S,q_T)$ (Ha)")
