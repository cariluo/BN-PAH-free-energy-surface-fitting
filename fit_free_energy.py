import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import gaussian_kde
from scipy.interpolate import RegularGridInterpolator

plt.rcParams["font.family"] = "Times New Roman"

# Parameters
T = 300.0
kB = 3.166811563e-6  # Hartree / K
kBT = kB * T
beta = 1.0 / kBT

N_SAMPLES = 13000
N_ACF = 80
fit_padding_factor = 0.0
plot_padding_factor = 0.5
q0_padding_factor = 0.0
n_grid = 50
N_GRID = 150
N_Q0 = 150

# Output directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Read energies
E_PBE_all = np.loadtxt("data/PBE0_energies.txt")[:N_SAMPLES]
E_PBE = E_PBE_all[::N_ACF]
qS_all = np.loadtxt("data/qS_energies.txt")[:N_SAMPLES]
qS = qS_all[::N_ACF]
qT_all = np.loadtxt("data/qT_energies.txt")[:N_SAMPLES]
qT = qT_all[::N_ACF]
S0_all = np.loadtxt("data/S0_energies.txt")[:N_SAMPLES]
S0 = S0_all[::N_ACF]

q0_all = S0_all - E_PBE_all
q0 = S0 - E_PBE

if not (len(E_PBE) == len(qS) == len(qT) == len(S0)):
    raise ValueError("Input files do not contain the same number of entries.")

samples_in_fit = len(q0)
print(f"Using {samples_in_fit} samples in the fitting")

idx_all = np.arange(N_SAMPLES)
idx = idx_all[::N_ACF]

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

q = np.column_stack((q0, qS, qT))

# Weighted empirical 2D free-energy surface
data = np.vstack([qS, qT])
weights = np.exp(-beta * q0)
weights /= weights.sum()
kde = gaussian_kde(data, weights=weights)

qS_grid = np.linspace(qS.min(), qS.max(), N_GRID)
qT_grid = np.linspace(qT.min(), qT.max(), N_GRID)
QS, QT = np.meshgrid(qS_grid, qT_grid)
positions = np.vstack([QS.ravel(), QT.ravel()])
P = kde(positions).reshape(QS.shape)
G_empirical = -kBT * np.log(P)
G_empirical -= G_empirical.min()

G_interpolator = RegularGridInterpolator(
    (qT_grid, qS_grid), G_empirical, bounds_error=False, fill_value=None
)
G_sampled = G_interpolator(np.column_stack([qT, qS]))

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

# Cubic polynomial basis
def polynomial_basis(q):
    """Return the 20-term cubic polynomial basis."""
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    return np.column_stack([
        np.ones(len(q)), x, y, z,
        x**2, y**2, z**2, x * y, x * z, y * z,
        x**3, y**3, z**3,
        x**2 * y, x**2 * z, y**2 * x, y**2 * z,
        z**2 * x, z**2 * y, x * y * z,
    ])

q_mean = np.mean(q, axis=0)
q_std = np.std(q, axis=0)
q_scaled = (q - q_mean) / q_std
X = polynomial_basis(q_scaled)

def polynomial(theta, X):
    return X @ theta

def free_energy(theta, X):
    return polynomial(theta, X) ** 2

# 3D integration grid for likelihood
q_min = np.min(q_scaled, axis=0)
q_max = np.max(q_scaled, axis=0)
padding = fit_padding_factor * (q_max - q_min)
box_min = q_min - padding
box_max = q_max + padding

q1_grid = np.linspace(box_min[0], box_max[0], n_grid)
q2_grid = np.linspace(box_min[1], box_max[1], n_grid)
q3_grid = np.linspace(box_min[2], box_max[2], n_grid)
dq1, dq2, dq3 = q1_grid[1] - q1_grid[0], q2_grid[1] - q2_grid[0], q3_grid[1] - q3_grid[0]

w1 = np.ones(n_grid) * dq1
w2 = np.ones(n_grid) * dq2
w3 = np.ones(n_grid) * dq3
w1[[0, -1]] *= 0.5
w2[[0, -1]] *= 0.5
w3[[0, -1]] *= 0.5

Q1, Q2, Q3 = np.meshgrid(q1_grid, q2_grid, q3_grid, indexing="ij")
q_integration = np.column_stack([Q1.ravel(), Q2.ravel(), Q3.ravel()])
X_integration = polynomial_basis(q_integration)
W1, W2, W3 = np.meshgrid(w1, w2, w3, indexing="ij")
integration_weights = (W1 * W2 * W3).ravel()
log_integration_weights = np.log(integration_weights)

def negative_log_likelihood(theta):
    G_data = free_energy(theta, X)
    G_grid = free_energy(theta, X_integration)
    if not np.all(np.isfinite(G_data)) or not np.all(np.isfinite(G_grid)):
        return np.inf
    log_Z = logsumexp(-beta * G_grid + log_integration_weights)
    if not np.isfinite(log_Z):
        return np.inf
    return beta * np.sum(G_data) + len(q_scaled) * log_Z

# Fit
theta0 = np.zeros(20)
theta0[0] = np.sqrt(kBT)

print("Starting optimization...")
print(f"Number of samples: {len(q)}")
print(f"Temperature:       {T:.2f} K")
print(f"kBT:               {kBT:.8e} Ha")
print(f"beta:              {beta:.8e} Ha^-1")

result = minimize(
    negative_log_likelihood,
    theta0,
    method="L-BFGS-B",
    options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-8, "maxls": 50},
)
print(result)
if not result.success:
    print("WARNING: optimization did not fully converge.")

theta = result.x
DeltaG_PBE = free_energy(theta, X)

output_data = np.column_stack([idx, q[:, 0], q[:, 1], q[:, 2], DeltaG_PBE])
np.savetxt(
    RESULTS_DIR / f"DeltaG_PBE_n_grid_{n_grid}_fit_padding_factor_{fit_padding_factor}.txt",
    output_data,
    fmt=["%d", "%.10e", "%.10e", "%.10e", "%.10e"],
    header="index q0 qS qT DeltaG_PBE_Hartree",
)

# Reweighted 2D surfaces
qS_min, qS_max = q[:, 1].min(), q[:, 1].max()
qT_min, qT_max = q[:, 2].min(), q[:, 2].max()
qS_pad = plot_padding_factor * (qS_max - qS_min)
qT_pad = plot_padding_factor * (qT_max - qT_min)
qS_grid = np.linspace(qS_min - qS_pad, qS_max, N_GRID)
qT_grid = np.linspace(qT_min - qT_pad, qT_max, N_GRID)

q0_min, q0_max = q[:, 0].min(), q[:, 0].max()
q0_pad = q0_padding_factor * (q0_max - q0_min)
q0_grid = np.linspace(q0_min - q0_pad, q0_max, N_Q0)
dq0 = q0_grid[1] - q0_grid[0]

QS, QT = np.meshgrid(qS_grid, qT_grid, indexing="ij")
G0_surface = np.zeros_like(QS)

for i in range(N_GRID):
    for j in range(N_GRID):
        points = np.column_stack([
            q0_grid,
            np.full(N_Q0, QS[i, j]),
            np.full(N_Q0, QT[i, j]),
        ])
        points_scaled = (points - q_mean) / q_std
        G_PBE = free_energy(theta, polynomial_basis(points_scaled))
        exponent = -beta * (G_PBE + q0_grid)
        log_integral = logsumexp(exponent) + np.log(dq0)
        G0_surface[i, j] = -kBT * log_integral

GS_surface = G0_surface + QS
GT_surface = G0_surface + QT
G0_min = np.min(G0_surface)
G0_surface -= G0_min
GS_surface -= G0_min
GT_surface -= G0_min

def calculate_G0(qS_values, qT_values):
    """Calculate unshifted G0(qS, qT) by numerical integration over q0."""
    qS_values = np.asarray(qS_values)
    qT_values = np.asarray(qT_values)
    if qS_values.shape != qT_values.shape:
        raise ValueError("qS_values and qT_values must have the same shape.")
    G0_values = np.empty(qS_values.size)
    for index, (qS_value, qT_value) in enumerate(zip(qS_values.ravel(), qT_values.ravel())):
        points = np.column_stack([
            q0_grid,
            np.full(N_Q0, qS_value),
            np.full(N_Q0, qT_value),
        ])
        points_scaled = (points - q_mean) / q_std
        G_PBE = free_energy(theta, polynomial_basis(points_scaled))
        exponent = -beta * (G_PBE + q0_grid)
        log_integral = logsumexp(exponent) + np.log(dq0)
        G0_values[index] = -kBT * log_integral
    return G0_values.reshape(qS_values.shape)

G0_sampled = calculate_G0(q[:, 1], q[:, 2])
GS_sampled = G0_sampled + q[:, 1]
GT_sampled = G0_sampled + q[:, 2]
G0_sampled -= G0_min
GS_sampled -= G0_min
GT_sampled -= G0_min

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
