from collections.abc import Iterable, Sequence
from typing import Any

from draive.generation.text.state import TextGeneration
from draive.instructions import Instruction
from draive.lmm import AnyTool, Toolbox
from draive.scope import ctx
from draive.types import MultimodalContent, MultimodalContentConvertible

__all__ = [
    "generate_text",
]


async def generate_text(
    *,
    instruction: Instruction | str,
    input: MultimodalContent | MultimodalContentConvertible,  # noqa: A002
    prefill: str | None = None,
    tools: Toolbox | Sequence[AnyTool] | None = None,
    examples: Iterable[tuple[MultimodalContent | MultimodalContentConvertible, str]] | None = None,
    **extra: Any,
) -> str:
    return await ctx.state(TextGeneration).generate(
        instruction=instruction,
        input=input,
        prefill=prefill,
        tools=tools,
        examples=examples,
        **extra,
    )
