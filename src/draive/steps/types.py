from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol, Self, runtime_checkable

from haiway import State

from draive.instructions import Instruction
from draive.lmm import AnyTool, LMMOutputSelection, Toolbox
from draive.multimodal import Multimodal, MultimodalContent

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
    def of(  # noqa: PLR0913
        cls,
        input: Multimodal,  # noqa: A002
        /,
        *,
        instruction: Instruction | str | None = None,
        tools: Toolbox | Iterable[AnyTool] | None = None,
        output: LMMOutputSelection = "auto",
        volatile: bool = False,
        inclusive: bool = False,
        condition: Callable[[], bool] | None = None,
        result_processing: StepResultProcessing | None = None,
        **extra: Any,
    ) -> Self:
        return cls(
            instruction=Instruction.of(instruction) if instruction else None,
            input=MultimodalContent.of(input),
            toolbox=Toolbox.of(tools),
            output=output,
            volatile=volatile,
            inclusive=inclusive,
            condition=condition,
            result_processing=result_processing,
            extra=extra,
        )

    instruction: Instruction | None
    input: MultimodalContent
    toolbox: Toolbox
    output: LMMOutputSelection
    volatile: bool
    inclusive: bool
    condition: Callable[[], bool] | None
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
