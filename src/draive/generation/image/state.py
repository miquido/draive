from typing import Any

from haiway import State, ctx

from draive.generation.image.typing import ImageGenerating
from draive.instructions import Instruction
from draive.multimodal import MediaContent, Multimodal

__all__ = ("ImageGeneration",)


class ImageGeneration(State):
    @classmethod
    async def generate(
        cls,
        *,
        instruction: Instruction | str,
        input: Multimodal | None = None,  # noqa: A002
        **extra: Any,
    ) -> MediaContent:
        return await ctx.state(ImageGeneration).generate(
            instruction=instruction,
            input=input,
            **extra,
        )

    generating: ImageGenerating
