from collections.abc import Iterable
from typing import Any

from draive.models import (
    GenerativeModel,
    ModelInput,
    ModelInstructions,
    ModelOutput,
    Toolbox,
)
from draive.multimodal import MultimodalContent

__all__ = ("generate_text",)


async def generate_text(
    *,
    instructions: ModelInstructions,
    input: MultimodalContent,  # noqa: A002
    toolbox: Toolbox,
    examples: Iterable[tuple[MultimodalContent, str]],
    **extra: Any,
) -> str:
    result: ModelOutput = await GenerativeModel.loop(
        instructions=instructions,
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
        stream=False,
        **extra,
    )

    return result.content.to_str()
