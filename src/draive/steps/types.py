from collections.abc import Iterable
from typing import Any, Literal, Protocol, Self, runtime_checkable

from haiway import State

from draive.instructions import Instruction
from draive.lmm import AnyTool, Toolbox
from draive.parameters import ParametersSpecification
from draive.types import Multimodal, MultimodalContent

__all__ = [
    "StepsCompleting",
    "Step",
]


class Step(State):
    @classmethod
    def of(
        cls,
        input: Multimodal,  # noqa: A002
        /,
        *,
        instruction: Instruction | str | None = None,
        tools: Toolbox | Iterable[AnyTool] | None = None,
        output: Literal["auto", "text"] | ParametersSpecification = "auto",
        **extra: Any,
    ) -> Self:
        return cls(
            instruction=Instruction.of(instruction) if instruction else None,
            input=MultimodalContent.of(input),
            toolbox=Toolbox.of(tools),
            output=output,
            extra=extra,
        )

    instruction: Instruction | None
    input: MultimodalContent
    toolbox: Toolbox
    output: Literal["auto", "text"] | ParametersSpecification
    extra: dict[str, Any]


@runtime_checkable
class StepsCompleting(Protocol):
    async def __call__(
        self,
        *steps: Step | Multimodal,
        instruction: Instruction | str | None = None,
        **extra: Any,
    ) -> MultimodalContent: ...
