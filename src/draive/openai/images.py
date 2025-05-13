from typing import Any

from haiway import ctx
from openai.types.image import Image
from openai.types.images_response import ImagesResponse

from draive.generation import ImageGeneration
from draive.instructions import Instruction
from draive.multimodal import (
    MediaContent,
    MediaData,
    MediaReference,
    Multimodal,
)
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIImageGenerationConfig
from draive.openai.types import OpenAIException
from draive.openai.utils import unwrap_missing

__all__ = ("OpenAIImageGeneration",)


class OpenAIImageGeneration(OpenAIAPI):
    def image_generation(self) -> ImageGeneration:
        return ImageGeneration(generating=self.generate_image)

    async def generate_image(
        self,
        *,
        instruction: Instruction | str,
        input: Multimodal | None,  # noqa: A002
        config: OpenAIImageGenerationConfig | None = None,
        **extra: Any,
    ) -> MediaContent:
        generation_config: OpenAIImageGenerationConfig = config or ctx.state(
            OpenAIImageGenerationConfig
        )
        with ctx.scope("generate_image", generation_config):
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

            if response.data is None:
                raise OpenAIException("Invalid OpenAI response - missing image content")

            image: Image = response.data[0]
            if url := image.url:
                return MediaReference.of(
                    url,
                    media="image/png",
                )

            elif b64data := image.b64_json:
                return MediaData.of(
                    b64data,
                    media="image/png",
                )

            else:
                raise OpenAIException("Invalid OpenAI response - missing image content")
