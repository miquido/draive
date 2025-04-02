from collections import deque
from collections.abc import AsyncIterator

__all__ = (
    "ConstantStream",
    "FixedStream",
)


class ConstantStream[Value](AsyncIterator[Value]):
    def __init__(
        self,
        value: Value,
        /,
    ) -> None:
        self._value: Value = value

    async def __anext__(self) -> Value:
        return self._value


class FixedStream[Value](AsyncIterator[Value]):
    def __init__(
        self,
        *values: Value,
    ) -> None:
        self._values: deque[Value] = deque(values)

    async def __anext__(self) -> Value:
        if self._values:
            return self._values.popleft()

        else:
            raise StopAsyncIteration()
