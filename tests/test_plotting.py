import matplotlib
from matplotlib.axes import Axes
import numpy as np

matplotlib.use("Agg")

from bn_pah_fes.plotting import (
    plot_likelihood,
    plot_likelihood_convergence,
    plot_singlet_triplet_gap_profiles,
)


def test_likelihood_plots_multiple_models(tmp_path):
    histories = {
        "harmonic": ([10.0, 4.0, 1.0], 1.0),
        "cubic": ([12.0, 3.0, 2.0], 2.0),
    }

    plot_likelihood(histories, tmp_path)
    plot_likelihood_convergence(histories, tmp_path)

    assert (tmp_path / "likelihood.png").is_file()
    assert (tmp_path / "likelihood_convergence.png").is_file()



def test_gap_profiles_mark_both_projected_minima(tmp_path, monkeypatch):
    qS_grid = np.linspace(-1.0, 1.0, 5)
    qT_grid = np.linspace(-1.0, 1.0, 5)
    GS_surface = (qS_grid[:, None] - 0.5) ** 2 + np.zeros_like(qT_grid)
    GT_surface = (qT_grid[None, :] + 0.5) ** 2 + np.zeros_like(qS_grid)[:, None]

    minimum_markers = []
    annotations = []
    original_text = Axes.text

    def capture_scatter(*args, **kwargs):
        if kwargs.get("marker") == "*":
            minimum_markers.append((args, kwargs))

    def capture_text(self, x, y, text, **kwargs):
        annotations.append((x, y, text, kwargs))
        return original_text(self, x, y, text, **kwargs)

    monkeypatch.setattr("bn_pah_fes.plotting.plt.scatter", capture_scatter)
    monkeypatch.setattr(Axes, "text", capture_text)

    plot_singlet_triplet_gap_profiles(
        qS_grid,
        qT_grid,
        GS_surface,
        GT_surface,
        beta=1.0,
        results_dir=tmp_path,
    )

    assert (tmp_path / "singlet_triplet_gap_profiles.png").is_file()
    assert len(minimum_markers) == 2
    assert all("label" not in marker[1] for marker in minimum_markers)
    assert len(annotations) == 1
    x, y, annotation, annotation_kwargs = annotations[0]
    assert (x, y) == (0.98, 0.98)
    assert annotation_kwargs["ha"] == "right"
    assert r"$G_S^{\min} - G_T^{\min} = $" in annotation
