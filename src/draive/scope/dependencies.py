from abc import ABC, abstractmethod
from asyncio import gather
from typing import Self, cast, final

from draive.scope.errors import MissingScopeDependency

__all__ = [
    "ScopeDependencies",
    "ScopeDependency",
]


class ScopeDependency(ABC):
    @classmethod
    def interface(cls) -> type:
        return cls

    @classmethod
    @abstractmethod
    def prepare(cls) -> Self: ...

    async def dispose(self) -> None:  # noqa: B027
        pass


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

    def dependency[Dependency_T: ScopeDependency](
        self,
        dependency: type[Dependency_T],
        /,
    ) -> Dependency_T:
        if dependency in self._dependencies:
            return cast(Dependency_T, self._dependencies[dependency])

        else:
            raise MissingScopeDependency(
                f"{dependency.__qualname__} is not defined!"
                " You have to define it when creating a new context."
            )

    async def dispose(self) -> None:
        await gather(*[dependency.dispose() for dependency in self._dependencies.values()])
