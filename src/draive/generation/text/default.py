from collections.abc import Iterable
from typing import Any

from draive.models import (
    ModelInput,
    ModelInstructions,
    ModelOutput,
)
from draive.multimodal import Multimodal, MultimodalContent
from draive.steps import Step
from draive.tools import Toolbox

__all__ = ("generate_text",)


async def generate_text(
    *,
    instructions: ModelInstructions,
    input: Multimodal,  # noqa: A002
    toolbox: Toolbox,
    examples: Iterable[tuple[Multimodal, str]],
    **extra: Any,
) -> str:
    completion: MultimodalContent = await Step.looping_completion(
        instructions=instructions,
        tools=toolbox,
        output="text",
        **extra,
    ).run(
        (
            *(
                message
                for example in examples
                for message in (
                    ModelInput.of(MultimodalContent.of(example[0])),
                    ModelOutput.of(MultimodalContent.of(example[1])),
                )
            ),
            ModelInput.of(MultimodalContent.of(input)),
        )
    )

    return completion.to_str()
