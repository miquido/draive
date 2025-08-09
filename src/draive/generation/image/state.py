from typing import Any, overload

from haiway import State, statemethod

from draive.generation.image.default import generate_image
from draive.generation.image.types import ImageGenerating
from draive.models import ResolveableInstructions
from draive.multimodal import MediaContent, Multimodal, MultimodalContent

__all__ = ("ImageGeneration",)


class ImageGeneration(State):
    @overload
    @classmethod
    async def generate(
        cls,
        *,
        instructions: ResolveableInstructions = "",
        input: Multimodal,
        **extra: Any,
    ) -> MediaContent: ...

    @overload
    async def generate(
        self,
        *,
        instructions: ResolveableInstructions = "",
        input: Multimodal,
        **extra: Any,
    ) -> MediaContent: ...

    @statemethod
    async def generate(
        self,
        *,
        instructions: ResolveableInstructions = "",
        input: Multimodal,  # noqa: A002
        **extra: Any,
    ) -> MediaContent:
        return await self.generating(
            instructions=instructions,
            input=MultimodalContent.of(input),
            **extra,
        )

    generating: ImageGenerating = generate_image
