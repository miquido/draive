from abc import ABC, abstractmethod
from asyncio import gather, shield
from types import TracebackType
from typing import Self, cast, final

from draive.scope.errors import MissingScopeDependency
from draive.utils import freeze

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
        *dependencies: type[ScopeDependency] | object,
    ) -> None:
        self._declared: tuple[type[ScopeDependency] | object, ...] = dependencies
        self._prepared: dict[type[object], object] | None = None

    @property
    def _dependencies(self) -> dict[type[object], object]:
        if self._prepared is None:
            dependencies: dict[type[object], object] = {}
            for dependency in self._declared:
                if isinstance(dependency, ScopeDependency):
                    dependencies[type(dependency).interface()] = dependency

                elif isinstance(dependency, type) and issubclass(dependency, ScopeDependency):
                    dependencies[dependency.interface()] = dependency.prepare()

                else:
                    dependencies[type(dependency)] = dependency

            self._prepared = dependencies
            del self._declared
            freeze(self)

        return self._prepared

    def dependency[Dependency](
        self,
        dependency: type[Dependency],
        /,
    ) -> Dependency:
        if dependency in self._dependencies:
            return cast(Dependency, self._dependencies[dependency])

        else:
            raise MissingScopeDependency(
                f"{dependency.__qualname__} is not defined or became disposed!"
                " You have to define it when creating a new context."
            )

    async def dispose(self) -> None:
        if self._prepared is None:
            # prevent preparing again
            self._prepared = {}
            del self._declared
            freeze(self)

        elif self._prepared:
            # avoid preparing when disposing
            await gather(
                *[
                    dependency.dispose()
                    for dependency in self._prepared.values()
                    if isinstance(dependency, ScopeDependency)
                ]
            )

            # cleanup memory and prevent preparing again
            # can't simply replace with new instance due to freeze
            for key in list(self._prepared.keys()):
                del self._prepared[key]

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await shield(self.dispose())
