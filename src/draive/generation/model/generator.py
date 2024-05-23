from collections.abc import Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

from draive.lmm import AnyTool, Toolbox
from draive.types import Instruction, Model, MultimodalContent, MultimodalContentElement

__all__ = [
    "ModelGenerator",
]


@runtime_checkable
class ModelGenerator(Protocol):
    async def __call__[Generated: Model](  # noqa: PLR0913
        self,
        generated: type[Generated],
        /,
        *,
        instruction: Instruction | str,
        input: MultimodalContent | MultimodalContentElement,  # noqa: A002
        schema_variable: str | None = None,
        tools: Toolbox | Sequence[AnyTool] | None = None,
        examples: Iterable[tuple[MultimodalContent | MultimodalContentElement, Generated]]
        | None = None,
        **extra: Any,
    ) -> Generated: ...
