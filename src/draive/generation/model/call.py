from collections.abc import Iterable, Sequence
from typing import Any

from draive.generation.model.state import ModelGeneration
from draive.lmm import AnyTool, Toolbox
from draive.scope import ctx
from draive.types import Instruction, Model, MultimodalContent, MultimodalContentElement

__all__ = [
    "generate_model",
]


async def generate_model[Generated: Model](  # noqa: PLR0913
    generated: type[Generated],
    /,
    *,
    instruction: Instruction | str,
    input: MultimodalContent | MultimodalContentElement,  # noqa: A002
    schema_variable: str | None = None,
    tools: Toolbox | Sequence[AnyTool] | None = None,
    examples: Iterable[tuple[MultimodalContent | MultimodalContentElement, Generated]]
    | None = None,
    **extra: Any,
) -> Generated:
    return await ctx.state(ModelGeneration).generate(
        generated,
        instruction=instruction,
        input=input,
        schema_variable=schema_variable,
        tools=tools,
        examples=examples,
        **extra,
    )
