from typing import Any, Protocol, runtime_checkable

from draive.models import ModelInstructions
from draive.multimodal import Multimodal
from draive.resources import ResourceContent, ResourceReference

__all__ = ("AudioGenerating",)


@runtime_checkable
class AudioGenerating(Protocol):
    async def __call__(
        self,
        *,
        instructions: ModelInstructions,
        input: Multimodal,  # noqa: A002
        **extra: Any,
    ) -> ResourceContent | ResourceReference: ...
