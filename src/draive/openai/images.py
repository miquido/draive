from typing import Any

from openai.types.image import Image

from draive.openai.client import OpenAIClient
from draive.openai.config import OpenAIImageGenerationConfig
from draive.openai.errors import OpenAIException
from draive.scope import ctx
from draive.types import ImageBase64Content, ImageContent, ImageURLContent

__all__ = [
    "openai_generate_image",
]


async def openai_generate_image(
    *,
    instruction: str,
    **extra: Any,
) -> ImageContent:
    client: OpenAIClient = ctx.dependency(OpenAIClient)
    config: OpenAIImageGenerationConfig = ctx.state(OpenAIImageGenerationConfig).updated(**extra)
    with ctx.nested("openai_generate_image", metrics=[config]):
        image: Image = await client.generate_image(
            config=config,
            instruction=instruction,
        )
        if url := image.url:
            return ImageURLContent(image_url=url)

        elif b64data := image.b64_json:
            return ImageBase64Content(image_base64=b64data)

        else:
            raise OpenAIException("Invalid OpenAI response - missing image content")
