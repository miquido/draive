from typing import Any

from haiway import ctx
from openai.types.image import Image

from draive.generation import ImageGenerator
from draive.instructions import Instruction
from draive.openai.client import SHARED, OpenAIClient
from draive.openai.config import OpenAIImageGenerationConfig
from draive.openai.types import OpenAIException
from draive.types import (
    ImageBase64Content,
    ImageContent,
    ImageURLContent,
    Multimodal,
)

__all__ = [
    "openai_image_generator",
]


async def openai_image_generator(
    client: OpenAIClient = SHARED,
    /,
) -> ImageGenerator:
    async def openai_generate_image(
        *,
        instruction: Instruction | str,
        input: Multimodal | None,  # noqa: A002
        **extra: Any,
    ) -> ImageContent:
        with ctx.scope("generate_image"):
            config: OpenAIImageGenerationConfig = ctx.state(OpenAIImageGenerationConfig).updated(
                **extra
            )
            image: Image = await client.generate_image(
                config=config,
                instruction=Instruction.formatted(instruction),
            )
            if url := image.url:
                return ImageURLContent(image_url=url)

            elif b64data := image.b64_json:
                return ImageBase64Content(image_base64=b64data)

            else:
                raise OpenAIException("Invalid OpenAI response - missing image content")

    return openai_generate_image
