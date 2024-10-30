from haiway import State

from draive.lmm.types import LMMInvocating, LMMStreaming

__all__: list[str] = [
    "LMMInvocating",
    "LMMStream",
]


class LMMInvocation(State):
    invoke: LMMInvocating


class LMMStream(State):
    prepare: LMMStreaming
