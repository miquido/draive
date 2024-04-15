from typing import Protocol, runtime_checkable

__all__ = [
    "UpdateSend",
]


@runtime_checkable
class UpdateSend[Value](Protocol):
    def __call__(
        self,
        update: Value,
    ): ...
