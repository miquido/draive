from typing import Any

from haiway import ctx
from openai.types.image import Image
from openai.types.images_response import ImagesResponse

from draive.generation import ImageGeneration
from draive.models import InstructionsRepository, ResolveableInstructions
from draive.multimodal import (
    MediaContent,
    MediaData,
    MediaReference,
    MultimodalContent,
)
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIImageGenerationConfig

__all__ = ("OpenAIImageGeneration",)


class OpenAIImageGeneration(OpenAIAPI):
    def image_generation(self) -> ImageGeneration:
        return ImageGeneration(generating=self.generate_image)

    async def generate_image(
        self,
        *,
        instructions: ResolveableInstructions,
        input: MultimodalContent,  # noqa: A002
        config: OpenAIImageGenerationConfig | None = None,
        **extra: Any,
    ) -> MediaContent:
        generation_config: OpenAIImageGenerationConfig = config or ctx.state(
            OpenAIImageGenerationConfig
        )
        async with ctx.scope("generate_image"):
            response: ImagesResponse = await self._client.images.generate(
                model=generation_config.model,
                n=1,
                prompt=await InstructionsRepository.resolve(instructions),
                quality=generation_config.quality,
                size=generation_config.size,
                style=generation_config.style,
                response_format=generation_config.result,
            )

            if response.data is None:
                raise ValueError("Invalid OpenAI response - missing image content")

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
                raise ValueError("Invalid OpenAI response - missing image content")
