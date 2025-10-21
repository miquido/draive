from typing import Any, overload

from haiway import State, statemethod

from draive.generation.audio.default import generate_audio
from draive.generation.audio.types import AudioGenerating
from draive.models import ModelInstructions
from draive.multimodal import Multimodal, MultimodalContent, Template
from draive.resources import ResourceContent, ResourceReference

__all__ = ("AudioGeneration",)


class AudioGeneration(State):
    @overload
    @classmethod
    async def generate(
        cls,
        *,
        instructions: Template | ModelInstructions = "",
        input: Multimodal,
        **extra: Any,
    ) -> ResourceContent | ResourceReference: ...

    @overload
    async def generate(
        self,
        *,
        instructions: Template | ModelInstructions = "",
        input: Multimodal,
        **extra: Any,
    ) -> ResourceContent | ResourceReference: ...

    @statemethod
    async def generate(
        self,
        *,
        instructions: Template | ModelInstructions = "",
        input: Multimodal,  # noqa: A002
        **extra: Any,
    ) -> ResourceContent | ResourceReference:
        return await self.generating(
            instructions=instructions,
            input=MultimodalContent.of(input),
            **extra,
        )

    generating: AudioGenerating = generate_audio
