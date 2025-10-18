from typing import Any

from haiway import ctx
from openai.types.image import Image
from openai.types.images_response import ImagesResponse

from draive.generation import ImageGeneration
from draive.models import ModelInstructions
from draive.multimodal import MultimodalContent, Template, TemplatesRepository
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIImageGenerationConfig
from draive.resources import ResourceContent, ResourceReference

__all__ = ("OpenAIImageGeneration",)


class OpenAIImageGeneration(OpenAIAPI):
    def image_generation(self) -> ImageGeneration:
        return ImageGeneration(generating=self.generate_image)

    async def generate_image(
        self,
        *,
        instructions: Template | ModelInstructions,
        input: MultimodalContent,  # noqa: A002
        config: OpenAIImageGenerationConfig | None = None,
        **extra: Any,
    ) -> ResourceContent | ResourceReference:
        generation_config: OpenAIImageGenerationConfig = config or ctx.state(
            OpenAIImageGenerationConfig
        )
        async with ctx.scope("generate_image"):
            response: ImagesResponse = await self._client.images.generate(
                model=generation_config.model,
                n=1,
                prompt=await TemplatesRepository.resolve_str(instructions),
                quality=generation_config.quality,
                size=generation_config.size,
                style=generation_config.style,
                response_format=generation_config.result,
            )

            if response.data is None:
                raise ValueError("Invalid OpenAI response - missing image content")

            image: Image = response.data[0]
            if url := image.url:
                return ResourceReference.of(
                    url,
                    mime_type="image/png",
                )

            elif b64data := image.b64_json:
                return ResourceContent.of(
                    b64data,
                    mime_type="image/png",
                )

            else:
                raise ValueError("Invalid OpenAI response - missing image content")
