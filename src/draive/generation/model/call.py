from collections.abc import Iterable
from typing import Any, Literal

from haiway import ctx

from draive.generation.model.state import ModelGeneration
from draive.generation.model.types import ModelGeneratorDecoder
from draive.instructions import Instruction
from draive.lmm import AnyTool, Toolbox
from draive.multimodal import Multimodal
from draive.parameters import DataModel

__all__ = [
    "generate_model",
]


async def generate_model[Generated: DataModel](  # noqa: PLR0913
    generated: type[Generated],
    /,
    *,
    instruction: Instruction | str,
    input: Multimodal,  # noqa: A002
    schema_injection: Literal["auto", "full", "simplified", "skip"] = "auto",
    tools: Toolbox | Iterable[AnyTool] | None = None,
    examples: Iterable[tuple[Multimodal, Generated]] | None = None,
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
