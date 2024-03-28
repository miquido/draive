from collections.abc import Iterable
from typing import Protocol, TypeVar, runtime_checkable

from draive.tools import Toolbox
from draive.types import Model, MultimodalContent

__all__ = [
    "ModelGenerator",
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
        input: MultimodalContent,  # noqa: A002
        tools: Toolbox | None = None,
        examples: Iterable[tuple[MultimodalContent, _Generated]] | None = None,
    ) -> _Generated:
        ...
