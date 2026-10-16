from bn_pah_fes.cli import choose_parameters


def test_choose_parameters_defaults_to_harmonic():
    output = []

    params = choose_parameters(
        input_fn=lambda prompt: "",
        output_fn=output.append,
    )

    assert params.model == "harmonic"
    assert params.initialization == "default"
    assert output[-2:] == [
        "Selected model: harmonic",
        "Initialization: default",
    ]


def test_choose_parameters_allows_squared_quadratic_plus_cubic_initialization():
    responses = iter(["5", "2"])
    output = []

    params = choose_parameters(
        input_fn=lambda prompt: next(responses),
        output_fn=output.append,
    )

    assert params.model == "squared_quadratic_plus_cubic"
    assert params.initialization == "squared_quadratic_fit"
    assert output[-2:] == [
        "Selected model: squared_quadratic_plus_cubic",
        "Initialization: squared_quadratic_fit",
    ]


def test_choose_parameters_reprompts_after_invalid_selection():
    responses = iter(["0", "2"])
    output = []

    params = choose_parameters(
        input_fn=lambda prompt: next(responses),
        output_fn=output.append,
    )

    assert params.model == "cubic"
    assert "Invalid choice: '0'. Enter a number from 1 to 5." in output
