from draive.choice.completion import ChoiceCompletion
from draive.choice.lmm import lmm_choice_completion
from draive.parameters import State

__all__: list[str] = [
    "Choice",
]


class Choice(State):
    completion: ChoiceCompletion = lmm_choice_completion
