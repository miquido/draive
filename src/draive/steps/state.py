from haiway import State

from draive.steps.default import default_steps_completion
from draive.steps.types import StepsCompleting

__all__: list[str] = [
    "Steps",
]


class Steps(State):
    completion: StepsCompleting = default_steps_completion
