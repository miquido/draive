from draive.lmm.invocation import LMMInvocation
from draive.parameters import State

__all__: list[str] = [
    "LMM",
]


class LMM(State):
    invocation: LMMInvocation
