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


def _choose_option(
    heading: str,
    choices: tuple[str, ...],
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str:
    """Prompt for one of ``choices``, defaulting to the first option."""
    output_fn(heading)
    for index, choice in enumerate(choices, start=1):
        output_fn(f"  {index}. {choice}")

    while True:
        selection = input_fn("Choice [1]: ").strip()
        if not selection:
            return choices[0]

        try:
            return choices[int(selection) - 1]
        except (IndexError, ValueError):
            output_fn(
                f"Invalid choice: {selection!r}. "
                f"Enter a number from 1 to {len(choices)}."
            )


def choose_parameters(
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> Parameters:
    """Interactively select a free-energy model and its initialization."""
    model = _choose_option(
        "Select free-energy model:",
        MODEL_CHOICES,
        input_fn,
        output_fn,
    )

    initialization = "default"
    if model == "squared_quadratic_plus_cubic":
        initialization = _choose_option(
            "Select initialization:",
            INITIALIZATION_CHOICES,
            input_fn,
            output_fn,
        )

    params = Parameters(model=model, initialization=initialization)
    output_fn(f"Selected model: {params.model}")
    output_fn(f"Initialization: {params.initialization}")
    return params
