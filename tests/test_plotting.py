import matplotlib

matplotlib.use("Agg")

from bn_pah_fes.plotting import plot_likelihood_convergence


def test_plot_likelihood_convergence_plots_multiple_models(tmp_path):
    plot_likelihood_convergence(
        {
            "harmonic": ([10.0, 4.0, 1.0], 1.0),
            "cubic": ([12.0, 3.0, 2.0], 2.0),
        },
        tmp_path,
    )

    assert (tmp_path / "likelihood_convergence.png").is_file()
