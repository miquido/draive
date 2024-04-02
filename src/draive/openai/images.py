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
) -> ImageContent:
    client: OpenAIClient = ctx.dependency(OpenAIClient)
    config: OpenAIImageGenerationConfig = ctx.state(OpenAIImageGenerationConfig)
    with ctx.nested("openai_generate_image", metrics=[config]):
        image: Image = await client.generate_image(
            config=config,
            instruction=instruction,
        )
        if url := image.url:
            return ImageURLContent(url=url)
        elif b64data := image.b64_json:
            return ImageBase64Content(base64=b64data)
        else:
            raise OpenAIException("Invalid OpenAI response - missing image content")
