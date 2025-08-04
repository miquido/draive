from collections.abc import Iterable
from typing import Any

from haiway import State, ctx

from draive.generation.text.default import generate_text
from draive.generation.text.types import TextGenerating
from draive.instructions import Instruction
from draive.multimodal import Multimodal, MultimodalContent
from draive.prompts import Prompt
from draive.tools import Tool, Toolbox

__all__ = ("TextGeneration",)


class TextGeneration(State):
    @classmethod
    async def generate(
        cls,
        *,
        instruction: Instruction | str | None = None,
        input: Prompt | Multimodal,  # noqa: A002
        tools: Toolbox | Iterable[Tool] | None = None,
        examples: Iterable[tuple[Multimodal, str]] | None = None,
        **extra: Any,
    ) -> str:
        return await ctx.state(cls).generation(
            instruction=Instruction.of(instruction),
            input=input if isinstance(input, Prompt) else MultimodalContent.of(input),
            toolbox=Toolbox.of(tools),
            examples=((MultimodalContent.of(example[0]), example[1]) for example in examples)
            if examples is not None
            else (),
            **extra,
        )

    generation: TextGenerating = generate_text
