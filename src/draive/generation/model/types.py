from collections.abc import Iterable
from typing import Any, Literal, Protocol, runtime_checkable

from draive.instructions import Instruction
from draive.multimodal import MultimodalContent
from draive.parameters import DataModel
from draive.prompts import Prompt
from draive.tools import Toolbox

__all__ = (
    "ModelGenerating",
    "ModelGeneratorDecoder",
)


@runtime_checkable
class ModelGeneratorDecoder(Protocol):
    def __call__(
        self,
        generated: MultimodalContent,
    ) -> dict[str, Any]: ...


@runtime_checkable
class ModelGenerating(Protocol):
    async def __call__[Generated: DataModel](
        self,
        generated: type[Generated],
        /,
        *,
        instruction: Instruction,
        input: Prompt | MultimodalContent,  # noqa: A002
        schema_injection: Literal["auto", "full", "simplified", "skip"],
        toolbox: Toolbox,
        examples: Iterable[tuple[MultimodalContent, Generated]] | None,
        decoder: ModelGeneratorDecoder | None,
        **extra: Any,
    ) -> Generated: ...
