from collections.abc import Iterable
from typing import Any

from haiway import ctx

from draive.choice.state import Choice
from draive.choice.types import ChoiceOption
from draive.instructions import Instruction
from draive.lmm import AnyTool, Toolbox
from draive.multimodal import Multimodal

__all__ = [
    "choice_completion",
]


async def choice_completion(
    *,
    instruction: Instruction | str,
    options: Iterable[ChoiceOption | Multimodal],
    input: Multimodal,  # noqa: A002
    tools: Toolbox | Iterable[AnyTool] | None = None,
    examples: Iterable[tuple[Multimodal, ChoiceOption]] | None = None,
    **extra: Any,
) -> ChoiceOption:
    return await ctx.state(Choice).completion(
        instruction=instruction,
        options=[
            option if isinstance(option, ChoiceOption) else ChoiceOption.of(option)
            for option in options
        ],
        input=input,
        toolbox=Toolbox.of(tools),
        examples=examples,
        **extra,
    )
