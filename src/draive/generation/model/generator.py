from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from draive.tools import Toolbox
from draive.types import Model, MultimodalContent

__all__ = [
    "ModelGenerator",
]


@runtime_checkable
class ModelGenerator(Protocol):
    async def __call__[Generated: Model](  # noqa: PLR0913
        self,
        model: type[Generated],
        *,
        instruction: str,
        input: MultimodalContent,  # noqa: A002
        tools: Toolbox | None = None,
        examples: Iterable[tuple[MultimodalContent, Generated]] | None = None,
    ) -> Generated: ...
