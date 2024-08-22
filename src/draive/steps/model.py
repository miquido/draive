from collections.abc import Sequence
from typing import Self

from draive.instructions import Instruction
from draive.lmm import AnyTool, Toolbox
from draive.parameters import State
from draive.types import MultimodalContent, MultimodalContentConvertible

__all__ = [
    "Step",
]


class Step(State):
    @classmethod
    def of(
        cls,
        input: MultimodalContent | MultimodalContentConvertible,  # noqa: A002
        /,
        *,
        instruction: Instruction | str | None = None,
        prefill: MultimodalContent | MultimodalContentConvertible | None = None,
        tools: Toolbox | Sequence[AnyTool] | None = None,
    ) -> Self:
        toolbox: Toolbox
        match tools:
            case None:
                toolbox = Toolbox()

            case Toolbox() as tools:
                toolbox = tools

            case sequence:
                toolbox = Toolbox(*sequence)

        return cls(
            instruction=Instruction.of(instruction) if instruction else None,
            input=MultimodalContent.of(input),
            prefill=prefill,
            toolbox=toolbox,
        )

    instruction: Instruction | None
    input: MultimodalContent
    prefill: MultimodalContent | MultimodalContentConvertible | None
    toolbox: Toolbox
