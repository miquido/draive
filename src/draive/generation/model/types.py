from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from haiway import State

from draive.models import ModelInstructions
from draive.multimodal import Multimodal, MultimodalContent
from draive.tools import Toolbox

__all__ = (
    "ModelGenerating",
    "ModelGenerationDecoder",
)


@runtime_checkable
class ModelGenerationDecoder[Generated: State](Protocol):
    def __call__(
        self,
        generated: MultimodalContent,
    ) -> Generated: ...


@runtime_checkable
class ModelGenerating(Protocol):
    async def __call__[Generated: State](
        self,
        generated: type[Generated],
        /,
        *,
        instructions: ModelInstructions,
        input: Multimodal,  # noqa: A002
        toolbox: Toolbox,
        examples: Iterable[tuple[Multimodal, Generated]],
        decoder: ModelGenerationDecoder[Generated] | None,
        **extra: Any,
    ) -> Generated: ...
