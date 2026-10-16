from bn_pah_fes.cli import choose_parameters


def test_choose_parameters_defaults_to_harmonic():
    output = []
    parameter_sets = choose_parameters(input_fn=lambda prompt: "", output_fn=output.append)

    assert len(parameter_sets) == 1
    assert parameter_sets[0].model == "harmonic"
    assert parameter_sets[0].initialization == "default"
    assert output[-3:] == [
        "Selected model: harmonic",
        "Initialization: default",
        "Fit padding factor: 1.0",
    ]


def test_choose_parameters_allows_multiple_models_and_initialization():
    responses = iter(["2,5", "2"])
    output = []
    parameter_sets = choose_parameters(
        input_fn=lambda prompt: next(responses), output_fn=output.append
    )

    assert [params.model for params in parameter_sets] == [
        "cubic", "squared_quadratic_plus_cubic"
    ]
    assert [params.initialization for params in parameter_sets] == [
        "default", "squared_quadratic_fit"
    ]


def test_choose_parameters_reprompts_after_invalid_or_duplicate_selection():
    responses = iter(["0", "2,2", "2,3"])
    output = []
    parameter_sets = choose_parameters(
        input_fn=lambda prompt: next(responses), output_fn=output.append
    )

    assert [params.model for params in parameter_sets] == ["cubic", "squared_quadratic"]
    assert "Invalid choices: '0'. Enter distinct numbers from 1 to 5, separated by commas." in output
    assert "Invalid choices: '2,2'. Enter distinct numbers from 1 to 5, separated by commas." in output
