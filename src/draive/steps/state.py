from draive.parameters import State
from draive.steps.completion import StepsCompletion
from draive.steps.lmm import lmm_steps_completion

__all__: list[str] = [
    "Steps",
]


class Steps(State):
    completion: StepsCompletion = lmm_steps_completion
