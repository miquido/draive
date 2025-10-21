from collections.abc import Iterable
from typing import Any, Literal, Protocol, runtime_checkable

from draive.models import ModelInstructions, Toolbox
from draive.multimodal import MultimodalContent, Template
from draive.parameters import DataModel

__all__ = (
    "ModelGenerating",
    "ModelGenerationDecoder",
)


@runtime_checkable
class ModelGenerationDecoder[Generated: DataModel](Protocol):
    def __call__(
        self,
        generated: MultimodalContent,
    ) -> Generated: ...


@runtime_checkable
class ModelGenerating(Protocol):
    async def __call__[Generated: DataModel](
        self,
        generated: type[Generated],
        /,
        *,
        instructions: Template | ModelInstructions,
        input: MultimodalContent,  # noqa: A002
        schema_injection: Literal["full", "simplified", "skip"],
        toolbox: Toolbox,
        examples: Iterable[tuple[MultimodalContent, Generated]],
        decoder: ModelGenerationDecoder[Generated] | None,
        **extra: Any,
    ) -> Generated: ...
