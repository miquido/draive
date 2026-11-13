from typing import Any, overload

from haiway import State, ctx, statemethod

from draive.generation.audio.default import generate_audio
from draive.generation.audio.types import AudioGenerating
from draive.models import ModelInstructions
from draive.multimodal import Multimodal, Template, TemplatesRepository
from draive.resources import ResourceContent, ResourceReference

__all__ = ("AudioGeneration",)


class AudioGeneration(State):
    @overload
    @classmethod
    async def generate(
        cls,
        *,
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,
        **extra: Any,
    ) -> ResourceContent | ResourceReference: ...

    @overload
    async def generate(
        self,
        *,
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,
        **extra: Any,
    ) -> ResourceContent | ResourceReference: ...

    @statemethod
    async def generate(
        self,
        *,
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,  # noqa: A002
        **extra: Any,
    ) -> ResourceContent | ResourceReference:
        """Generate audio using the configured generative model.

        Parameters
        ----------
        instructions : Template | ModelInstructions, default=""
            Instructions or a template resolved before generating.
        input : Template | Multimodal
            Input content or a template resolved before generating.
        **extra : Any
            Additional provider options forwarded to generation.

        Returns
        -------
        ResourceContent | ResourceReference
            Generated audio content or a reference to it.
        """
        async with ctx.scope("audio_generation"):
            if isinstance(instructions, Template):
                ctx.record_info(
                    attributes={"instructions.template": instructions.identifier},
                )
                instructions = await TemplatesRepository.resolve_str(instructions)

            if isinstance(input, Template):
                ctx.record_info(
                    attributes={"input.template": input.identifier},
                )
                input = await TemplatesRepository.resolve(input)  # noqa: A001

            return await self._generating(
                instructions=instructions,
                input=input,
                **extra,
            )

    _generating: AudioGenerating = generate_audio

    def __init__(
        self,
        generating: AudioGenerating = generate_audio,
    ) -> None:
        super().__init__(_generating=generating)
