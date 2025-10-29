from typing import Any, overload

from haiway import State, statemethod

from draive.generation.image.default import generate_image
from draive.generation.image.types import ImageGenerating
from draive.models import ModelInstructions
from draive.multimodal import Multimodal, Template, TemplatesRepository
from draive.resources import ResourceContent, ResourceReference

__all__ = ("ImageGeneration",)


class ImageGeneration(State):
    @overload
    @classmethod
    async def generate(
        cls,
        *,
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,
        **extra: Any,
    ) -> ResourceContent | ResourceReference: ...

    @overload
    async def generate(
        self,
        *,
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,
        **extra: Any,
    ) -> ResourceContent | ResourceReference: ...

    @statemethod
    async def generate(
        self,
        *,
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,  # noqa: A002
        **extra: Any,
    ) -> ResourceContent | ResourceReference:
        return await self.generating(
            # resolve instructions templates
            instructions=await TemplatesRepository.resolve_str(instructions),
            # resolve input templates
            input=await TemplatesRepository.resolve(input),
            **extra,
        )

    generating: ImageGenerating = generate_image
