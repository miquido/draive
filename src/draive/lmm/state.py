from draive.lmm.completion import LMMCompletion
from draive.types import State

__all__: list[str] = [
    "LMM",
]


class LMM(State):
    completion: LMMCompletion
