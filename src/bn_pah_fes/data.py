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


def load_data(data_dir: Path, n_samples: int, n_acf: int) -> EnergyData:
    """Load input energy trajectories and construct fitting coordinates."""
    E_pbe_all = np.loadtxt(data_dir / "PBE0_energies.txt")[:n_samples]
    E_pbe = E_pbe_all[::n_acf]

    qS_all = np.loadtxt(data_dir / "qS_energies.txt")[:n_samples]
    qS = qS_all[::n_acf]

    qT_all = np.loadtxt(data_dir / "qT_energies.txt")[:n_samples]
    qT = qT_all[::n_acf]

    S0_all = np.loadtxt(data_dir / "S0_energies.txt")[:n_samples]
    S0 = S0_all[::n_acf]

    q0_all = S0_all - E_pbe_all
    q0 = S0 - E_pbe

    if not (len(E_pbe) == len(qS) == len(qT) == len(S0)):
        raise ValueError("Input files do not contain the same number of entries.")

    idx_all = np.arange(n_samples)
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
