from abc import ABC, abstractmethod
from asyncio import gather
from contextvars import ContextVar, Token
from types import TracebackType
from typing import Self, TypeVar, cast, final

from draive.scope.errors import MissingScopeDependency

__all__ = [
    "ScopeDependencies",
    "ScopeDependency",
    "_ScopeDependency_T",
]


class ScopeDependency(ABC):
    @classmethod
    def interface(cls) -> type:
        return cls

    @classmethod
    @abstractmethod
    def prepare(cls) -> Self:
        ...

    async def dispose(self) -> None:  # noqa: B027
        pass


_ScopeDependency_T = TypeVar(
    "_ScopeDependency_T",
    bound=ScopeDependency,
)


@final
class ScopeDependencies:
    def __init__(
        self,
        *dependencies: ScopeDependency | type[ScopeDependency],
    ) -> None:
        self._dependencies: dict[type[ScopeDependency], ScopeDependency] = {}
        for dependency in dependencies:
            if isinstance(dependency, ScopeDependency):
                self._dependencies[type(dependency).interface()] = dependency
            else:
                self._dependencies[dependency.interface()] = dependency.prepare()
        self._token: Token[ScopeDependencies] | None = None

    def dependency(
        self,
        _type: type[_ScopeDependency_T],
        /,
    ) -> _ScopeDependency_T:
        if _type in self._dependencies:
            return cast(_ScopeDependency_T, self._dependencies[_type])

        else:
            raise MissingScopeDependency(
                f"{_type} is not defined! You have to define it when creating context."
            )

    async def dispose(self) -> None:
        await gather(*[dependency.dispose() for dependency in self._dependencies.values()])

    def __enter__(self) -> None:
        assert self._token is None, "Reentrance is not allowed"  # nosec: B101
        self._token = _ScopeDependencies_Var.set(self)

    def __exit__(
        self,
        exc_type: BaseException | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._token is None:
            raise AttributeError("Can't exit scope without entering")
        _ScopeDependencies_Var.reset(self._token)

    async def __aenter__(self) -> None:
        assert self._token is None, "Reentrance is not allowed"  # nosec: B101
        self._token = _ScopeDependencies_Var.set(self)

    async def __aexit__(
        self,
        exc_type: BaseException | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._token is None:
            raise AttributeError("Can't exit scope without entering")
        _ScopeDependencies_Var.reset(self._token)
        await self.dispose()


_ScopeDependencies_Var = ContextVar[ScopeDependencies]("_ScopeDependencies_Var")
