from collections.abc import Callable

from .config import Parameters


MODEL_CHOICES = (
    "harmonic",
    "cubic",
    "squared_quadratic",
    "squared_cubic",
    "squared_quadratic_plus_cubic",
)
INITIALIZATION_CHOICES = ("default", "squared_quadratic_fit")


def _choose_options(
    heading: str,
    choices: tuple[str, ...],
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> tuple[str, ...]:
    """Prompt for one or more numbered choices, defaulting to the first."""
    output_fn(heading)
    for index, choice in enumerate(choices, start=1):
        output_fn(f"  {index}. {choice}")

    while True:
        selection = input_fn("Choices [1]: ").strip()
        if not selection:
            return (choices[0],)

        try:
            indices = [int(item.strip()) for item in selection.split(",")]
            if not indices or len(set(indices)) != len(indices):
                raise ValueError
            return tuple(choices[index - 1] for index in indices)
        except (IndexError, ValueError):
            output_fn(
                f"Invalid choices: {selection!r}. "
                f"Enter distinct numbers from 1 to {len(choices)}, separated by commas."
            )


def _choose_option(
    heading: str,
    choices: tuple[str, ...],
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str:
    """Prompt for one choice, defaulting to the first option."""
    return _choose_options(heading, choices, input_fn, output_fn)[0]


def choose_parameters(
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> tuple[Parameters, ...]:
    """Interactively select one or more free-energy-model runs."""
    models = _choose_options(
        "Select free-energy model(s):",
        MODEL_CHOICES,
        input_fn,
        output_fn,
    )

    parameter_sets = []
    for model in models:
        initialization = "default"
        if model == "squared_quadratic_plus_cubic":
            initialization = _choose_option(
                "Select initialization:",
                INITIALIZATION_CHOICES,
                input_fn,
                output_fn,
            )

        params = Parameters(model=model, initialization=initialization)
        parameter_sets.append(params)
        output_fn(f"Selected model: {params.model}")
        output_fn(f"Initialization: {params.initialization}")
        output_fn(f"Fit padding factor: {params.fit_padding_factor}")

    return tuple(parameter_sets)
