from collections.abc import Iterable
from typing import Any

from haiway import State, ctx

from draive.lmm import LMMContext
from draive.multimodal import Multimodal, MultimodalContent
from draive.parameters.model import DataModel
from draive.prompts import Prompt
from draive.utils.processing import Processing
from draive.workflow.stage import Stage

__all__ = [
    "workflow_completion",
]


async def workflow_completion(
    stage: Stage | Prompt | Multimodal,
    /,
    *stages: Stage | Prompt | Multimodal,
    state: Iterable[State] | None = None,
    **extra: Any,
) -> MultimodalContent:
    workflow_state = WorkflowState(state)
    async with ctx.scope(
        "workflow_completion",
        Processing(
            # keep current event reporting unchanged
            event_reporting=ctx.state(Processing).event_reporting,
            # but use local state for workflow
            state_reading=workflow_state.read,
            state_writing=workflow_state.write,
        ),
    ):
        current_context: LMMContext = []
        current_result = MultimodalContent.of()
        for current in (stage, *stages):
            current_stage: Stage
            match current:
                case Stage() as stage:
                    current_stage = stage

                case Prompt() as prompt:
                    current_stage = Stage.completion(prompt)

                case multimodal:
                    current_stage = Stage.completion(multimodal)

            current_context, current_result = await current_stage._processing(
                context=current_context,
                result=current_result,
            )

        return current_result


class WorkflowState:
    def __init__(
        self,
        state: Iterable[DataModel | State] | None,
    ) -> None:
        self._state: dict[type[DataModel | State], Any]
        object.__setattr__(
            self,
            "_state",
            {type(element): element for element in state} if state else {},
        )

    async def read[StateType: DataModel | State](
        self,
        state: type[StateType],
        /,
        default: StateType | None = None,
    ) -> StateType | None:
        if state in self._state:
            return self._state[state]

        else:
            return default

    async def write(
        self,
        state: DataModel | State,
        /,
    ) -> None:
        self._state[type(state)] = state

    def __setattr__(
        self,
        __name: str,
        __value: Any,
    ) -> None:
        raise RuntimeError("WorkflowState is frozen and can't be modified")

    def __delattr__(
        self,
        __name: str,
    ) -> None:
        raise RuntimeError("WorkflowState is frozen and can't be modified")
