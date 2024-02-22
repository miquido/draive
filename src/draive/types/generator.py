from collections.abc import Iterable
from typing import Protocol, TypeVar

from draive.types.generated import Generated
from draive.types.string import StringConvertible
from draive.types.toolset import Toolset

__all__ = [
    "ModelGenerator",
    "TextGenerator",
]


_Generated = TypeVar(
    "_Generated",
    bound=Generated,
)


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
