from collections.abc import Sequence
from typing import Self

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
        input: MultimodalContentConvertible,  # noqa: A002
        /,
        *,
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
            input=MultimodalContent.of(input),
            toolbox=toolbox,
        )

    input: MultimodalContent
    toolbox: Toolbox
