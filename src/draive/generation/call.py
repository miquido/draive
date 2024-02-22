from collections.abc import Iterable
from typing import TypeVar

from draive.generation.state import ModelGeneration, TextGeneration
from draive.scope import ctx
from draive.types import Generated, StringConvertible, Toolset

__all__ = [
    "generate_text",
    "generate_model",
]


async def generate_text(
    *,
    instruction: str,
    input: StringConvertible,  # noqa: A002
    toolset: Toolset | None = None,
    examples: Iterable[tuple[str, str]] | None = None,
) -> str:
    text_generation: TextGeneration = ctx.state(TextGeneration)
    return await text_generation.generate(
        instruction=instruction,
        input=input,
        toolset=toolset or text_generation.toolset,
        examples=examples,
    )


_Generated = TypeVar(
    "_Generated",
    bound=Generated,
)


async def generate_model(
    model: type[_Generated],
    *,
    instruction: str,
    input: StringConvertible,  # noqa: A002
    toolset: Toolset | None = None,
    examples: Iterable[tuple[str, _Generated]] | None = None,
) -> _Generated:
    model_generation: ModelGeneration = ctx.state(ModelGeneration)
    return await model_generation.generate(
        model,
        instruction=instruction,
        input=input,
        toolset=toolset or model_generation.toolset,
        examples=examples,
    )
