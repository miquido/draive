from draive.parameters.state import State

__all__ = [
    "Embedded",
]


class Embedded[Value](State):
    value: Value
    vector: list[float]
