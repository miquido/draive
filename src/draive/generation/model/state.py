from collections.abc import Iterable
from typing import Any, Literal

from haiway import State, ctx

from draive.generation.model.default import generate_model
from draive.generation.model.types import ModelGenerating, ModelGeneratorDecoder
from draive.instructions import Instruction
from draive.multimodal import Multimodal, MultimodalContent
from draive.parameters import DataModel
from draive.prompts import Prompt
from draive.tools import Tool, Toolbox

__all__ = ("ModelGeneration",)


class ModelGeneration(State):
    @classmethod
    async def generate[Generated: DataModel](
        cls,
        generated: type[Generated],
        /,
        *,
        instruction: Instruction | str,
        input: Prompt | Multimodal,  # noqa: A002
        schema_injection: Literal["auto", "full", "simplified", "skip"] = "auto",
        tools: Toolbox | Iterable[Tool] | None = None,
        examples: Iterable[tuple[Multimodal, Generated]] | None = None,
        decoder: ModelGeneratorDecoder | None = None,
        **extra: Any,
    ) -> Generated:
        return await ctx.state(cls).generating(
            generated,
            instruction=Instruction.of(instruction),
            schema_injection=schema_injection,
            input=input if isinstance(input, Prompt) else MultimodalContent.of(input),
            toolbox=Toolbox.of(tools),
            examples=((MultimodalContent.of(example[0]), example[1]) for example in examples)
            if examples is not None
            else (),
            decoder=decoder,
            **extra,
        )

    generating: ModelGenerating = generate_model
