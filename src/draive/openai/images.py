from typing import Any

from haiway import ctx
from openai.types.image import Image

from draive.generation import ImageGenerator
from draive.instructions import Instruction
from draive.multimodal import (
    MediaContent,
    Multimodal,
)
from draive.openai.client import OpenAIClient
from draive.openai.config import OpenAIImageGenerationConfig
from draive.openai.types import OpenAIException

__all__ = [
    "openai_image_generator",
]


async def openai_image_generator(
    client: OpenAIClient | None = None,
    /,
) -> ImageGenerator:
    client = client or OpenAIClient.shared()

    async def openai_generate_image(
        *,
        instruction: Instruction | str,
        input: Multimodal | None,  # noqa: A002
        **extra: Any,
    ) -> MediaContent:
        with ctx.scope("generate_image"):
            config: OpenAIImageGenerationConfig = ctx.state(OpenAIImageGenerationConfig).updated(
                **extra
            )
            image: Image = await client.generate_image(
                config=config,
                instruction=Instruction.formatted(instruction),
            )
            if url := image.url:
                return MediaContent.url(
                    url,
                    media="image/png",
                )

            elif b64data := image.b64_json:
                return MediaContent.base64(
                    b64data,
                    media="image/png",
                )

            else:
                raise OpenAIException("Invalid OpenAI response - missing image content")

    return openai_generate_image
