from collections.abc import Iterable, Sequence
from typing import Any

from draive.choice.model import ChoiceOption
from draive.choice.state import Choice
from draive.instructions import Instruction
from draive.lmm import AnyTool, Toolbox
from draive.scope import ctx
from draive.types import Multimodal

__all__ = [
    "choice_completion",
]


async def choice_completion(  # noqa: PLR0913
    *,
    instruction: Instruction | str,
    options: Iterable[ChoiceOption | Multimodal],
    input: Multimodal,  # noqa: A002
    prefill: str | None = None,
    tools: Toolbox | Sequence[AnyTool] | None = None,
    examples: Iterable[tuple[Multimodal, ChoiceOption]] | None = None,
    **extra: Any,
) -> ChoiceOption:
    toolbox: Toolbox
    match tools:
        case None:
            toolbox = Toolbox()

        case Toolbox() as tools:
            toolbox = tools

        case sequence:
            toolbox = Toolbox(*sequence)

    return await ctx.state(Choice).completion(
        instruction=instruction,
        options=[
            option if isinstance(option, ChoiceOption) else ChoiceOption.of(option)
            for option in options
        ],
        input=input,
        prefill=prefill,
        toolbox=toolbox,
        examples=examples,
        **extra,
    )
