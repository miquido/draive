from haiway import State

from draive.choice.default import default_choice_completion
from draive.choice.types import ChoiceCompletion

__all__: list[str] = [
    "Choice",
]


class Choice(State):
    completion: ChoiceCompletion = default_choice_completion
