from collections.abc import Iterable
from typing import TypeVar

from draive.generation.model.state import ModelGeneration
from draive.scope import ctx
from draive.tools import Toolbox
from draive.types import Model, MultimodalContent

__all__ = [
    "generate_model",
]

_Generated = TypeVar(
    "_Generated",
    bound=Model,
)


async def generate_model(
    model: type[_Generated],
    *,
    instruction: str,
    input: MultimodalContent,  # noqa: A002
    tools: Toolbox | None = None,
    examples: Iterable[tuple[MultimodalContent, _Generated]] | None = None,
) -> _Generated:
    model_generation: ModelGeneration = ctx.state(ModelGeneration)
    return await model_generation.generate(
        model,
        instruction=instruction,
        input=input,
        tools=tools or model_generation.tools,
        examples=examples,
    )
