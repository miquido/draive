from typing import Any, Protocol, runtime_checkable

from draive.models import ResolveableInstructions
from draive.multimodal import MediaContent, MultimodalContent

__all__ = ("AudioGenerating",)


@runtime_checkable
class AudioGenerating(Protocol):
    async def __call__(
        self,
        *,
        instructions: ResolveableInstructions,
        input: MultimodalContent,  # noqa: A002
        **extra: Any,
    ) -> MediaContent: ...
