from typing import Any

from haiway import ctx, unwrap_missing
from openai import omit
from openai.types.image import Image
from openai.types.images_response import ImagesResponse

from draive.models import ModelInstructions
from draive.multimodal import Multimodal, MultimodalContent, Template, TemplatesRepository
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIImageGenerationConfig
from draive.resources import ResourceContent, ResourceReference

__all__ = ("OpenAIImageGeneration",)


class OpenAIImageGeneration(OpenAIAPI):
    async def generate_image(
        self,
        *,
        instructions: Template | ModelInstructions,
        input: Multimodal,  # noqa: A002
        config: OpenAIImageGenerationConfig | None = None,
        **extra: Any,
    ) -> ResourceContent | ResourceReference:
        generation_config: OpenAIImageGenerationConfig = config or ctx.state(
            OpenAIImageGenerationConfig
        )
        async with ctx.scope("generation.image.invocation"):
            # the image API takes a single prompt, the input content carries the subject
            # while instructions carry the guidance - both have to reach it
            content: MultimodalContent = MultimodalContent.of(input)
            if content.contains_resources:
                ctx.log_warning(
                    "OpenAI image generation does not accept resource input,"
                    " only its text is included in the prompt"
                )

            prompt: str = "\n\n".join(
                element
                for element in (
                    await TemplatesRepository.resolve_str(instructions),
                    content.to_str(),
                )
                if element
            )

            response: ImagesResponse = await self._client.images.generate(
                model=generation_config.model,
                n=1,
                prompt=prompt,
                quality=generation_config.quality,
                size=generation_config.size,
                background=generation_config.background,
                output_format=generation_config.output_format,
                output_compression=unwrap_missing(
                    generation_config.output_compression,
                    default=omit,
                ),
                moderation=generation_config.moderation,
            )

            if not response.data:
                raise ValueError("Invalid OpenAI response - missing image content")

            mime_type: str = f"image/{generation_config.output_format}"
            image: Image = response.data[0]
            if b64data := image.b64_json:
                return ResourceContent.of(
                    b64data,
                    mime_type=mime_type,
                )

            elif url := image.url:
                return ResourceReference.of(
                    url,
                    mime_type=mime_type,
                )

            else:
                raise ValueError("Invalid OpenAI response - missing image content")
