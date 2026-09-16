# BN-PAH Free-Energy Surface Fitting

Analysis and fitting tools for constructing free-energy surfaces in the `(q_S, q_T)` coordinate space for BN-PAH excited-state simulations.

## Workflow

`fit_free_energy.py` performs the following steps:

1. Reads PBE0, S0, S1, and T1-related energy-coordinate trajectories.
2. Constructs `q0 = S0 - PBE0` and subsamples the trajectory using the specified ACF interval.
3. Builds a reweighted 2D KDE estimate of the empirical free-energy surface.
4. Fits a squared cubic polynomial representation of the 3D free energy in `(q0, qS, qT)` using an L-BFGS-B likelihood optimization.
5. Numerically integrates over `q0` to obtain `ΔG0(qS,qT)`.
6. Constructs the singlet and triplet surfaces using
   `ΔGS = ΔG0 + qS` and `ΔGT = ΔG0 + qT`.
7. Writes diagnostic data and 2D/3D surface plots.

## Input files

The script expects these files in the working directory:

```text
PBE0_energies.txt
qS_energies.txt
qT_energies.txt
S0_energies.txt
```

Each file should contain one energy value per trajectory step, with matching lengths.

## Dependencies

The current script requires:

- Python 3
- NumPy
- SciPy
- Matplotlib

A minimal environment can be installed with:

```bash
pip install numpy scipy matplotlib
```

## Running

From the directory containing the four input files:

```bash
python fit_free_energy.py
```

The script produces diagnostic and surface files including:

```text
KDE.png
coordinate_time_series.png
coordinate_vector_distribution.png
DeltaG_PBE0.png
DeltaG0_contour.png
DeltaGS_contour.png
DeltaGT_contour.png
DeltaG0_surface.png
DeltaGS_surface.png
DeltaGT_surface.png
```

## Notes

The current implementation uses a 20-term cubic polynomial in scaled `(q0, qS, qT)` coordinates and represents the fitted PBE0 free energy as the square of that polynomial. The fitted 3D surface is subsequently reweighted by `exp(-beta*q0)` and integrated numerically over `q0` to construct the 2D surfaces.

This repository contains the analysis code; large trajectory/energy datasets are intentionally not included.
