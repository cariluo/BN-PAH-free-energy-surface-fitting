from dataclasses import dataclass


@dataclass(frozen=True)
class Parameters:
    """Parameters controlling the free-energy surface calculation."""

    temperature: float = 300.0
    kB: float = 3.166811563e-6  # Hartree / K
    n_samples: int = 13000
    n_acf: int = 80
    fit_padding_factor: float = 0.0
    plot_padding_factor: float = 0.0
    q0_padding_factor: float = 0.0
    n_grid: int = 50
    n_grid_surface: int = 150
    n_q0: int = 150
    model: str = "squared_cubic"

    @property
    def kBT(self) -> float:
        return self.kB * self.temperature

    @property
    def beta(self) -> float:
        return 1.0 / self.kBT
