from collections.abc import Iterable
from typing import Any, Literal, Protocol, runtime_checkable

from draive.instructions import Instruction
from draive.multimodal import Multimodal, MultimodalContent
from draive.parameters import DataModel
from draive.prompts import Prompt
from draive.tools import AnyTool, Toolbox

__all__ = [
    "ModelGenerator",
    "ModelGeneratorDecoder",
]


@runtime_checkable
class ModelGeneratorDecoder(Protocol):
    def __call__(
        self,
        generated: MultimodalContent,
    ) -> dict[str, Any]: ...


@runtime_checkable
class ModelGenerator(Protocol):
    async def __call__[Generated: DataModel](  # noqa: PLR0913
        self,
        generated: type[Generated],
        /,
        *,
        instruction: Instruction | str,
        input: Prompt | Multimodal,  # noqa: A002
        schema_injection: Literal["auto", "full", "simplified", "skip"],
        tools: Toolbox | Iterable[AnyTool] | None,
        examples: Iterable[tuple[Multimodal, Generated]] | None,
        decoder: ModelGeneratorDecoder | None,
        **extra: Any,
    ) -> Generated: ...
