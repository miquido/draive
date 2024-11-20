from collections.abc import Iterable, Mapping
from typing import Any, Literal, Protocol, Self, runtime_checkable

from haiway import State

from draive.instructions import Instruction
from draive.lmm import AnyTool, Toolbox
from draive.multimodal import Multimodal, MultimodalContent
from draive.parameters import ParametersSpecification

__all__ = [
    "StepsCompleting",
    "Step",
]


@runtime_checkable
class StepResultProcessing(Protocol):
    async def __call__(
        self,
        content: MultimodalContent,
    ) -> MultimodalContent: ...


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
        result_processing: StepResultProcessing | None = None,
        **extra: Any,
    ) -> Self:
        return cls(
            instruction=Instruction.of(instruction) if instruction else None,
            input=MultimodalContent.of(input),
            toolbox=Toolbox.of(tools),
            output=output,
            result_processing=result_processing,
            extra=extra,
        )

    instruction: Instruction | None
    input: MultimodalContent
    toolbox: Toolbox
    output: Literal["auto", "text"] | ParametersSpecification
    result_processing: StepResultProcessing | None
    extra: Mapping[str, Any]


@runtime_checkable
class StepsCompleting(Protocol):
    async def __call__(
        self,
        *steps: Step | Multimodal,
        instruction: Instruction | str | None = None,
        **extra: Any,
    ) -> MultimodalContent: ...
