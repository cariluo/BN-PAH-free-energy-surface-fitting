from bn_pah_fes.config import Parameters


def test_default_plot_padding_factor_is_nonzero() -> None:
    """The default plotting grid includes an extrapolation buffer."""
    assert Parameters().plot_padding_factor == 0.2
