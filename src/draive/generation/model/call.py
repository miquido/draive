from collections.abc import Iterable, Sequence
from typing import Any, Literal

from draive.generation.model.generator import ModelGeneratorDecoder
from draive.generation.model.state import ModelGeneration
from draive.lmm import AnyTool, Toolbox
from draive.parameters import DataModel
from draive.scope import ctx
from draive.types import Instruction, MultimodalContent, MultimodalContentConvertible

__all__ = [
    "generate_model",
]


async def generate_model[Generated: DataModel](  # noqa: PLR0913
    generated: type[Generated],
    /,
    *,
    instruction: Instruction | str,
    input: MultimodalContent | MultimodalContentConvertible,  # noqa: A002
    schema_injection: Literal["auto", "full", "simplified", "skip"] = "auto",
    tools: Toolbox | Sequence[AnyTool] | None = None,
    examples: Iterable[tuple[MultimodalContent | MultimodalContentConvertible, Generated]]
    | None = None,
    decoder: ModelGeneratorDecoder | None = None,
    **extra: Any,
) -> Generated:
    return await ctx.state(ModelGeneration).generate(
        generated,
        instruction=instruction,
        input=input,
        schema_injection=schema_injection,
        tools=tools,
        examples=examples,
        decoder=decoder,
        **extra,
    )
