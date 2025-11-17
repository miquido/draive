from collections.abc import Iterable
from typing import Any, overload

from haiway import State, ctx, statemethod

from draive.generation.text.default import generate_text
from draive.generation.text.types import TextGenerating
from draive.models import ModelInstructions, Tool, Toolbox
from draive.multimodal import Multimodal, MultimodalContent, Template, TemplatesRepository

__all__ = ("TextGeneration",)


class TextGeneration(State):
    @overload
    @classmethod
    async def generate(
        cls,
        *,
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,
        tools: Toolbox | Iterable[Tool] = (),
        examples: Iterable[tuple[Multimodal, str]] = (),
        **extra: Any,
    ) -> str: ...

    @overload
    async def generate(
        self,
        *,
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,
        tools: Toolbox | Iterable[Tool] = (),
        examples: Iterable[tuple[Multimodal, str]] = (),
        **extra: Any,
    ) -> str: ...

    @statemethod
    async def generate(
        self,
        *,
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,  # noqa: A002
        tools: Toolbox | Iterable[Tool] = (),
        examples: Iterable[tuple[Multimodal, str]] = (),
        **extra: Any,
    ) -> str:
        async with ctx.scope("generate_text"):
            if isinstance(instructions, Template):
                ctx.record_info(
                    attributes={"instructions.template": instructions.identifier},
                )

            if isinstance(input, Template):
                ctx.record_info(
                    attributes={"input.template": input.identifier},
                )

            return await self.generating(
                # resolve instructions templates
                instructions=await TemplatesRepository.resolve_str(instructions),
                # resolve input templates
                input=await TemplatesRepository.resolve(input),
                toolbox=Toolbox.of(tools),
                examples=((MultimodalContent.of(ex_in), ex_out) for ex_in, ex_out in examples),
                **extra,
            )

    generating: TextGenerating = generate_text
