from collections.abc import Iterable
from typing import Protocol, TypeVar, runtime_checkable

from draive.types.model import Model
from draive.types.string import StringConvertible
from draive.types.toolset import Toolset

__all__ = [
    "ModelGenerator",
    "TextGenerator",
]


_Generated = TypeVar(
    "_Generated",
    bound=Model,
)


@runtime_checkable
class ModelGenerator(Protocol):
    async def __call__(  # noqa: PLR0913
        self,
        model: type[_Generated],
        *,
        instruction: str,
        input: StringConvertible,  # noqa: A002
        toolset: Toolset | None = None,
        examples: Iterable[tuple[str, _Generated]] | None = None,
    ) -> _Generated:
        ...


@runtime_checkable
class TextGenerator(Protocol):
    async def __call__(
        self,
        *,
        instruction: str,
        input: StringConvertible,  # noqa: A002
        toolset: Toolset | None = None,
        examples: Iterable[tuple[str, str]] | None = None,
    ) -> str:
        ...
