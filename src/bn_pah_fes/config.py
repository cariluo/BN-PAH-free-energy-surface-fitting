from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Parameters:
    """Parameters controlling the free-energy surface calculation."""

    temperature: float = 300.0
    kB: float = 3.166811563e-6  # Hartree / K
    n_samples: int = 30000
    n_stride: int = 775
    fit_padding_factor: float = 1.0
    plot_padding_factor: float = 0.2
    q0_padding_factor: float = 0.0
    n_grid: int = 50
    n_grid_surface: int = 150
    n_q0: int = 150
    model: Literal[
        "squared_cubic",
        "squared_quadratic",
        "squared_quadratic_plus_cubic",
        "cubic",
        "harmonic",
    ] = "squared_cubic"
    initialization: Literal["default", "squared_quadratic_fit"] = "default"

    @property
    def kBT(self) -> float:
        return self.kB * self.temperature

    @property
    def beta(self) -> float:
        return 1.0 / self.kBT
