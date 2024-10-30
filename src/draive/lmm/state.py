from haiway import State

from draive.lmm.types import LMMInvocation

__all__: list[str] = [
    "LMM",
]


class LMM(State):
    invocation: LMMInvocation
