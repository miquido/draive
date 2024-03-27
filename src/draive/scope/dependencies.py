from abc import ABC, abstractmethod
from asyncio import gather
from contextvars import Token
from typing import Self, TypeVar, cast, final

from draive.scope.errors import MissingScopeDependency

__all__ = [
    "DependenciesScope",
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
class DependenciesScope:
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
        self._token: Token[DependenciesScope] | None = None

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
