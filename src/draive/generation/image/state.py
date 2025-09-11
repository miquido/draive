from typing import Any, overload

from haiway import State, statemethod

from draive.generation.image.default import generate_image
from draive.generation.image.types import ImageGenerating
from draive.models import ResolveableInstructions
from draive.multimodal import Multimodal, MultimodalContent
from draive.resources import ResourceContent, ResourceReference

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
    ) -> ResourceContent | ResourceReference: ...

    @overload
    async def generate(
        self,
        *,
        instructions: ResolveableInstructions = "",
        input: Multimodal,
        **extra: Any,
    ) -> ResourceContent | ResourceReference: ...

    @statemethod
    async def generate(
        self,
        *,
        instructions: ResolveableInstructions = "",
        input: Multimodal,  # noqa: A002
        **extra: Any,
    ) -> ResourceContent | ResourceReference:
        return await self.generating(
            instructions=instructions,
            input=MultimodalContent.of(input),
            **extra,
        )

    generating: ImageGenerating = generate_image
