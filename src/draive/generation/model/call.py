from collections.abc import Iterable
from typing import Any

from draive.generation.model.state import ModelGeneration
from draive.scope import ctx
from draive.tools import Toolbox
from draive.types import Model, MultimodalContent

__all__ = [
    "generate_model",
]


async def generate_model[Generated: Model](
    generated: type[Generated],
    /,
    *,
    instruction: str,
    input: MultimodalContent,  # noqa: A002
    tools: Toolbox | None = None,
    examples: Iterable[tuple[MultimodalContent, Generated]] | None = None,
    **extra: Any,
) -> Generated:
    model_generation: ModelGeneration = ctx.state(ModelGeneration)
    return await model_generation.generate(
        generated,
        instruction=instruction,
        input=input,
        tools=tools or model_generation.tools,
        examples=examples,
        **extra,
    )
