from collections.abc import Iterable
from typing import Protocol, TypeVar, runtime_checkable

from draive.types.model import Model
from draive.types.multimodal import MultimodalContent
from draive.types.toolset import Toolset

__all__ = [
    "ModelGenerator",
    "TextGenerator",
]


@runtime_checkable
class TextGenerator(Protocol):
    async def __call__(
        self,
        *,
        instruction: str,
        input: MultimodalContent,  # noqa: A002
        toolset: Toolset | None = None,
        examples: Iterable[tuple[MultimodalContent, str]] | None = None,
    ) -> str:
        ...


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
        input: MultimodalContent,  # noqa: A002
        toolset: Toolset | None = None,
        examples: Iterable[tuple[MultimodalContent, _Generated]] | None = None,
    ) -> _Generated:
        ...
