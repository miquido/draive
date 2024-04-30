from asyncio import Lock
from collections.abc import Iterable
from types import TracebackType
from typing import Any, Self
from weakref import proxy

from draive.helpers import freeze
from draive.parameters import ParametrizedData
from draive.tools import Tool, tool
from draive.types import (
    MultimodalContent,
    State,
    merge_multimodal_content,
    multimodal_content_string,
)

__all__ = [
    "AgentsChatMessage",
    "AgentsChat",
    "AgentsData",
    "AgentsDataAccess",
]


class AgentsChatMessage(State):
    author: str
    content: MultimodalContent

    def as_str(self) -> str:
        return f"{self.author}:\n{multimodal_content_string(self.content)}"

    def __str__(self) -> str:
        return self.as_str()


class AgentsChat(State):
    goal: MultimodalContent
    messages: tuple[AgentsChatMessage, ...] = ()

    def as_multimodal(self) -> MultimodalContent:
        return merge_multimodal_content(
            "Goal:",
            self.goal,
            *[
                merge_multimodal_content(
                    f"{message.author}:",
                    message.content,
                )
                for message in self.messages
            ],
        )

    def as_str(self) -> str:
        result: str = f"Goal: {multimodal_content_string(self.goal)}"
        for message in self.messages:
            result += f"\n---\n{message.as_str()}"
        return result

    def __str__(self) -> str:
        return self.as_str()

    def appending(
        self,
        message: AgentsChatMessage | None,
    ) -> Self:
        match message:
            case None:
                return self

            case message:
                return self.__class__(
                    goal=self.goal,
                    messages=(*self.messages, message),
                )


class AgentsData[Data: ParametrizedData]:
    def __init__(
        self,
        initial: Data,
    ) -> None:
        self._lock: Lock = Lock()
        self._data_type: type[Data] = type(initial)
        self._data: dict[str, object] = initial.as_dict()
        self._mutable_proxy: AgentsDataAccess[Data] | None = None
        self.access: AgentsDataAccess[Data] = AgentsDataAccess(source=self)

        freeze(self)

    @property
    async def current_data(self) -> Data:
        async with self._lock:
            return self._data_type.from_dict(self._data)

    @property
    async def current_contents(self) -> dict[str, object]:
        async with self._lock:
            return self._data.copy()

    def list_tool(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Tool[[], Any]:
        @tool(name=name or "list_data", description=description or "List all available data keys")
        async def data_list() -> Any:
            async with self._lock:
                return self.access.keys()

        return data_list

    def read_tool(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Tool[[str], Any]:
        @tool(
            name=name or "access_data", description=description or "Access the data under given key"
        )
        async def read_data(key: str) -> Any:
            async with self._lock:
                return self.access.__getattr__(name=key)

        return read_data

    def write_tool(
        self,
        name: str | None = None,
        description: str | None = None,
        key: str | None = None,
    ) -> Tool[[str, MultimodalContent], str] | Tool[[MultimodalContent], str]:
        if key := key:

            @tool(name=name or "save_data", description=description or "Save the data")
            async def write_specific(value: MultimodalContent) -> str:
                async with self._lock:
                    self.access.__setattr__(name=key, value=value)
                return "Saved!"

            return write_specific
        else:

            @tool(
                name=name or "save_data", description=description or "Save the data under given key"
            )
            async def write_any(key: str, value: MultimodalContent) -> str:
                async with self._lock:
                    self.access.__setattr__(name=key, value=value)
                return "Saved!"

            return write_any


class AgentsDataAccess[Data: ParametrizedData]:
    def __init__(
        self,
        source: AgentsData[Data],
    ) -> None:
        self._source: AgentsData[Data]
        object.__setattr__(self, "_source", proxy(source))

    @property
    def current(self) -> Data:
        assert self._source._lock.locked, "Can't use data access outside of lock"  # nosec: B101 # pyright: ignore[reportPrivateUsage]
        return self._source._data_type.from_dict(self._source._data)  # pyright: ignore[reportPrivateUsage]

    def set_current(
        self,
        data: Data,
        /,
    ) -> None:
        assert self._source._lock.locked, "Can't use data access outside of lock"  # nosec: B101 # pyright: ignore[reportPrivateUsage]
        self._source._data.update(**data.as_dict())  # pyright: ignore[reportPrivateUsage]

    def keys(self) -> Iterable[str]:
        assert self._source._lock.locked, "Can't use data access outside of lock"  # nosec: B101 # pyright: ignore[reportPrivateUsage]
        return self._source._data.keys()  # pyright: ignore[reportPrivateUsage]

    def __getattr__(self, name: str) -> Any | None:
        assert self._source._lock.locked, "Can't use data access outside of lock"  # nosec: B101 # pyright: ignore[reportPrivateUsage]
        return self._source._data.get(name)  # pyright: ignore[reportPrivateUsage]

    def __setattr__(self, name: str, value: Any) -> None:
        assert self._source._lock.locked, "Can't use data access outside of lock"  # nosec: B101 # pyright: ignore[reportPrivateUsage]
        self._source._data[name] = value  # pyright: ignore[reportPrivateUsage]

    def __delattr__(self, name: str) -> None:
        assert self._source._lock.locked, "Can't use data access outside of lock"  # nosec: B101 # pyright: ignore[reportPrivateUsage]
        del self._source._data[name]  # pyright: ignore[reportPrivateUsage]

    async def __aenter__(self) -> Self:
        await self._source._lock.__aenter__()  # pyright: ignore[reportPrivateUsage]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._source._lock.__aexit__(  # pyright: ignore[reportPrivateUsage]
            exc_type,
            exc_val,
            exc_tb,
        )
