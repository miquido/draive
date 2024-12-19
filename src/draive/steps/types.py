from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol, Self, overload, runtime_checkable

from haiway import State

from draive.instructions import Instruction
from draive.lmm import AnyTool, LMMCompletion, LMMOutputSelection, Toolbox
from draive.multimodal import Multimodal, MultimodalContent

__all__ = [
    "Step",
    "StepsCompleting",
]


@runtime_checkable
class StepResultProcessing(Protocol):
    async def __call__(
        self,
        content: MultimodalContent,
    ) -> MultimodalContent: ...


class Step(State):
    @overload
    @classmethod
    def of(
        cls,
        input: Multimodal,
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
        input: Multimodal,
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
    def of(  # noqa: PLR0913
        cls,
        input: Multimodal,  # noqa: A002
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

        return cls(
            instruction=Instruction.of(instruction) if instruction else None,
            input=MultimodalContent.of(input),
            toolbox=Toolbox.out_of(tools),
            output=output,
            completion=None if completion is None else LMMCompletion.of(completion),
            volatile=volatile,
            extends_result=extends_result,
            condition=condition,
            result_processing=result_processing,
            extra=extra,
        )

    instruction: Instruction | None
    input: MultimodalContent
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
        *steps: Step | Multimodal,
        instruction: Instruction | str | None = None,
        **extra: Any,
    ) -> MultimodalContent: ...
