from collections.abc import Iterable

from draive.generation.text.state import TextGeneration
from draive.scope import ctx
from draive.tools import Toolbox
from draive.types import MultimodalContent

__all__ = [
    "generate_text",
]


async def generate_text(
    *,
    instruction: str,
    input: MultimodalContent,  # noqa: A002
    tools: Toolbox | None = None,
    examples: Iterable[tuple[MultimodalContent, str]] | None = None,
) -> str:
    text_generation: TextGeneration = ctx.state(TextGeneration)
    return await text_generation.generate(
        instruction=instruction,
        input=input,
        tools=tools or text_generation.tools,
        examples=examples,
    )
