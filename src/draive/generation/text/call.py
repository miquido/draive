from collections.abc import Iterable
from typing import Any

from haiway import ctx

from draive.generation.text.state import TextGeneration
from draive.instructions import Instruction
from draive.multimodal import Multimodal
from draive.prompts import Prompt
from draive.tools import Tool, Toolbox

__all__ = ("generate_text",)


async def generate_text(
    *,
    instruction: Instruction | str | None = None,
    input: Prompt | Multimodal,  # noqa: A002
    tools: Toolbox | Iterable[Tool] | None = None,
    examples: Iterable[tuple[Multimodal, str]] | None = None,
    **extra: Any,
) -> str:
    return await ctx.state(TextGeneration).generate(
        instruction=instruction,
        input=input,
        tools=tools,
        examples=examples,
        **extra,
    )
