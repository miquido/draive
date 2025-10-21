from collections.abc import Iterable
from typing import Any

from haiway import ctx

from draive.models import (
    GenerativeModel,
    ModelInput,
    ModelInstructions,
    ModelOutput,
    Toolbox,
)
from draive.multimodal import MultimodalContent, Template, TemplatesRepository

__all__ = ("generate_text",)


async def generate_text(
    *,
    instructions: Template | ModelInstructions,
    input: MultimodalContent,  # noqa: A002
    toolbox: Toolbox,
    examples: Iterable[tuple[MultimodalContent, str]],
    **extra: Any,
) -> str:
    async with ctx.scope("generate_text"):
        result: ModelOutput = await GenerativeModel.loop(
            instructions=await TemplatesRepository.resolve_str(instructions),
            context=[
                *[
                    message
                    for example in examples
                    for message in [
                        ModelInput.of(example[0]),
                        ModelOutput.of(MultimodalContent.of(example[1])),
                    ]
                ],
                ModelInput.of(input),
            ],
            toolbox=toolbox,
            output="text",
            **extra,
        )

        return result.content.to_str()
