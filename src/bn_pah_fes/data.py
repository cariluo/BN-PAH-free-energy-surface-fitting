from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EnergyData:
    """Energy-coordinate trajectories used for fitting."""

    E_pbe_all: np.ndarray
    E_pbe: np.ndarray
    qS_all: np.ndarray
    qS: np.ndarray
    qT_all: np.ndarray
    qT: np.ndarray
    S0_all: np.ndarray
    S0: np.ndarray
    q0_all: np.ndarray
    q0: np.ndarray
    idx_all: np.ndarray
    idx: np.ndarray

    @property
    def samples_in_fit(self) -> int:
        return len(self.q0)

    @property
    def q(self) -> np.ndarray:
        return np.column_stack((self.q0, self.qS, self.qT))


def _load_energy_file(data_dir: Path, filename: str, n_samples: int) -> np.ndarray:
    """Load an energy trajectory and ensure it contains enough samples."""
    values = np.atleast_1d(np.loadtxt(data_dir / filename))
    if len(values) < n_samples:
        raise ValueError(
            f"{filename} contains {len(values)} entries, but "
            f"n_samples={n_samples} was requested."
        )
    return values[:n_samples]


def load_data(data_dir: Path, n_samples: int, n_acf: int) -> EnergyData:
    """Load input energy trajectories and construct fitting coordinates."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if n_acf <= 0:
        raise ValueError("n_acf must be positive.")

    E_pbe_all = _load_energy_file(data_dir, "PBE0_energies.txt", n_samples)
    qS_all = _load_energy_file(data_dir, "qS_energies.txt", n_samples)
    qT_all = _load_energy_file(data_dir, "qT_energies.txt", n_samples)
    S0_all = _load_energy_file(data_dir, "S0_energies.txt", n_samples)

    E_pbe = E_pbe_all[::n_acf]
    qS = qS_all[::n_acf]
    qT = qT_all[::n_acf]
    S0 = S0_all[::n_acf]

    q0_all = S0_all - E_pbe_all
    q0 = S0 - E_pbe

    idx_all = np.arange(len(E_pbe_all))
    idx = idx_all[::n_acf]

    return EnergyData(
        E_pbe_all=E_pbe_all,
        E_pbe=E_pbe,
        qS_all=qS_all,
        qS=qS,
        qT_all=qT_all,
        qT=qT,
        S0_all=S0_all,
        S0=S0,
        q0_all=q0_all,
        q0=q0,
        idx_all=idx_all,
        idx=idx,
    )
