from typing import Any

from haiway import ctx
from openai.types.image import Image
from openai.types.images_response import ImagesResponse

from draive.generation import ImageGeneration
from draive.instructions import Instruction
from draive.multimodal import (
    MediaContent,
    Multimodal,
)
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIImageGenerationConfig
from draive.openai.types import OpenAIException
from draive.openai.utils import unwrap_missing

__all__ = [
    "OpenAIImageGeneration",
]


class OpenAIImageGeneration(OpenAIAPI):
    def image_generation(self) -> ImageGeneration:
        return ImageGeneration(generate=self.generate_image)

    async def generate_image(
        self,
        *,
        instruction: Instruction | str,
        input: Multimodal | None,  # noqa: A002
        config: OpenAIImageGenerationConfig | None = None,
        **extra: Any,
    ) -> MediaContent:
        with ctx.scope("generate_image"):
            generation_config: OpenAIImageGenerationConfig = config or ctx.state(
                OpenAIImageGenerationConfig
            ).updated(**extra)

            response: ImagesResponse = await self._client.images.generate(
                model=generation_config.model,
                n=1,
                prompt=Instruction.formatted(instruction),
                quality=generation_config.quality,
                size=generation_config.size,
                style=generation_config.style,
                timeout=unwrap_missing(generation_config.timeout),
                response_format=generation_config.result,
            )

            image: Image = response.data[0]
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
