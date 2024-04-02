from draive.generation.image.state import ImageGeneration
from draive.scope import ctx
from draive.types import ImageContent

__all__ = [
    "generate_image",
]


async def generate_image(
    *,
    instruction: str,
) -> ImageContent:
    return await ctx.state(ImageGeneration).generate(
        instruction=instruction,
    )
