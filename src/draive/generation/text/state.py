from collections.abc import Iterable
from typing import Any, overload

from haiway import State, statemethod

from draive.generation.text.default import generate_text
from draive.generation.text.types import TextGenerating
from draive.models import ResolveableInstructions, Tool, Toolbox
from draive.multimodal import Multimodal, MultimodalContent

__all__ = ("TextGeneration",)


class TextGeneration(State):
    @overload
    @classmethod
    async def generate(
        cls,
        *,
        instructions: ResolveableInstructions = "",
        input: Multimodal,
        tools: Toolbox | Iterable[Tool] = (),
        examples: Iterable[tuple[Multimodal, str]] = (),
        **extra: Any,
    ) -> str: ...

    @overload
    async def generate(
        self,
        *,
        instructions: ResolveableInstructions = "",
        input: Multimodal,
        tools: Toolbox | Iterable[Tool] = (),
        examples: Iterable[tuple[Multimodal, str]] = (),
        **extra: Any,
    ) -> str: ...

    @statemethod
    async def generate(
        self,
        *,
        instructions: ResolveableInstructions = "",
        input: Multimodal,  # noqa: A002
        tools: Toolbox | Iterable[Tool] = (),
        examples: Iterable[tuple[Multimodal, str]] = (),
        **extra: Any,
    ) -> str:
        return await self.generating(
            instructions=instructions,
            input=MultimodalContent.of(input),
            toolbox=Toolbox.of(tools),
            examples=((MultimodalContent.of(ex_in), ex_out) for ex_in, ex_out in examples),
            **extra,
        )

    generating: TextGenerating = generate_text
