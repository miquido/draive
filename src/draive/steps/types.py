from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol, Self, overload, runtime_checkable

from haiway import State
from typing_extensions import deprecated

from draive.instructions import Instruction
from draive.lmm import LMMCompletion, LMMInput, LMMOutputSelection
from draive.multimodal import Multimodal, MultimodalContent
from draive.prompts import Prompt
from draive.tools import AnyTool, Toolbox

__all__ = (
    "Step",
    "StepsCompleting",
)


@runtime_checkable
class StepResultProcessing(Protocol):
    async def __call__(
        self,
        content: MultimodalContent,
    ) -> MultimodalContent: ...


@deprecated("`Step` has been replaced with `Stage`")
class Step(State):
    @overload
    @classmethod
    def of(
        cls,
        input: Prompt | Multimodal,
        /,
        *,
        completion: Multimodal,
        volatile: bool = False,
        extends_result: bool = False,
        condition: Callable[[], bool] | None = None,
        **extra: Any,
    ) -> Self: ...

    @overload
    @classmethod
    def of(
        cls,
        input: Prompt | Multimodal,
        /,
        *,
        instruction: Instruction | str | None = None,
        tools: Toolbox | Iterable[AnyTool] | None = None,
        output: LMMOutputSelection = "auto",
        volatile: bool = False,
        extends_result: bool = False,
        condition: Callable[[], bool] | None = None,
        result_processing: StepResultProcessing | None = None,
        **extra: Any,
    ) -> Self: ...

    @classmethod
    def of(
        cls,
        input: Prompt | Multimodal,  # noqa: A002
        /,
        *,
        instruction: Instruction | str | None = None,
        tools: Toolbox | Iterable[AnyTool] | None = None,
        output: LMMOutputSelection = "auto",
        completion: Multimodal | None = None,
        volatile: bool = False,
        extends_result: bool = False,
        condition: Callable[[], bool] | None = None,
        result_processing: StepResultProcessing | None = None,
        **extra: Any,
    ) -> Self:
        assert (  # nosec: B101
            completion is None or instruction is None
        ), "Can't specify instruction with predefined result"
        assert (  # nosec: B101
            completion is None or tools is None
        ), "Can't specify tools with predefined result"
        assert (  # nosec: B101
            completion is None or result_processing is None
        ), "Can't specify result processing with predefined result"

        step_input: Prompt | LMMInput
        match input:
            case Prompt() as prompt:
                step_input = prompt

            case other:
                step_input = LMMInput.of(other)

        return cls(
            instruction=Instruction.of(instruction),
            input=step_input,
            toolbox=Toolbox.of(tools),
            output=output,
            completion=None if completion is None else LMMCompletion.of(completion),
            volatile=volatile,
            extends_result=extends_result,
            condition=condition,
            result_processing=result_processing,
            extra=extra,
        )

    instruction: Instruction | None
    input: Prompt | LMMInput
    toolbox: Toolbox
    output: LMMOutputSelection
    completion: LMMCompletion | None
    volatile: bool
    extends_result: bool
    condition: Callable[[], bool] | None
    result_processing: StepResultProcessing | None
    extra: Mapping[str, Any]


@runtime_checkable
class StepsCompleting(Protocol):
    async def __call__(
        self,
        *steps: Step | Prompt | Multimodal,
        instruction: Instruction | str | None = None,
        **extra: Any,
    ) -> MultimodalContent: ...
