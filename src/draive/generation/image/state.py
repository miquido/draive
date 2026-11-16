from typing import Any, overload

from haiway import State, ctx, statemethod

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
        """Generate image using the configured generative model.

        Parameters
        ----------
        instructions : Template | ModelInstructions, default=""
            Instructions or a template resolved before generating.
        input : Template | Multimodal
            Input content or a template resolved before generating.
        **extra : Any
            Additional provider options forwarded to generation.

        Returns
        -------
        ResourceContent | ResourceReference
            Generated image content or a reference to it.
        """
        async with ctx.scope("image_generation"):
            if isinstance(instructions, Template):
                ctx.record_info(
                    attributes={"instructions.template": instructions.identifier},
                )
                instructions = await TemplatesRepository.resolve_str(instructions)

            if isinstance(input, Template):
                ctx.record_info(
                    attributes={"input.template": input.identifier},
                )
                input = await TemplatesRepository.resolve(input)  # noqa: A001

            return await self._generating(
                instructions=instructions,
                input=input,
                **extra,
            )

    _generating: ImageGenerating = generate_image

    def __init__(
        self,
        generating: ImageGenerating = generate_image,
    ) -> None:
        super().__init__(_generating=generating)
