from typing import Any

from draive.generation.image.state import ImageGeneration
from draive.scope import ctx
from draive.types import ImageContent, Instruction, MultimodalContent, MultimodalContentConvertible

__all__ = [
    "generate_image",
]


async def generate_image(
    *,
    instruction: Instruction | str,
    input: MultimodalContent | MultimodalContentConvertible | None = None,  # noqa: A002
    **extra: Any,
) -> ImageContent:
    return await ctx.state(ImageGeneration).generate(
        instruction=instruction,
        input=input,
        **extra,
    )
