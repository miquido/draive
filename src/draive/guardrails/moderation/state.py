from asyncio import ALL_COMPLETED, FIRST_COMPLETED, Task, wait
from collections.abc import (
    AsyncIterable,
    Collection,
    Coroutine,
)
from typing import Any, overload

from haiway import State, statemethod

from draive.guardrails.moderation.types import (
    GuardrailsInputModerationException,
    GuardrailsModerationChecking,
    GuardrailsModerationException,
    GuardrailsOutputModerationException,
)
from draive.guardrails.types import GuardrailsException, GuardrailsFailure
from draive.multimodal import Multimodal, MultimodalContent

__all__ = ("GuardrailsModeration",)


async def _no_moderation(
    content: MultimodalContent,
    /,
    **extra: Any,
) -> None:
    pass


class GuardrailsModeration(State):
    @overload
    @classmethod
    async def check_input(
        cls,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> None: ...

    @overload
    async def check_input(
        self,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> None: ...

    @statemethod
    async def check_input(
        self,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> None:
        content = MultimodalContent.of(content)
        try:
            await self.input_checking(
                content,
                **extra,
            )

        except GuardrailsInputModerationException:
            raise

        except GuardrailsModerationException as exc:
            raise GuardrailsInputModerationException(
                f"Input moderation guardrails triggered: {exc}",
                content=content,
                violations=exc.violations,
                replacement=exc.replacement,
                meta=exc.meta,
            ) from exc

        except GuardrailsException as exc:
            raise GuardrailsInputModerationException(
                f"Input moderation guardrails triggered: {exc}",
                content=content,
                violations={str(exc): 1.0},
                meta=exc.meta,
            ) from exc

        except Exception as exc:
            raise GuardrailsFailure(
                f"Input moderation guardrails failed: {exc}",
                cause=exc,
                meta={"error_type": exc.__class__.__name__},
            ) from exc

    @overload
    @classmethod
    async def input_guarded[Result](
        cls,
        content: Multimodal,
        /,
        feature: Coroutine[None, None, Result],
        **extra: Any,
    ) -> Result: ...

    @overload
    async def input_guarded[Result](
        self,
        content: Multimodal,
        /,
        feature: Coroutine[None, None, Result],
        **extra: Any,
    ) -> Result: ...

    @statemethod
    async def input_guarded[Result](
        self,
        content: Multimodal,
        /,
        feature: Coroutine[None, None, Result],
        **extra: Any,
    ) -> Result:
        feature_task: Task[Result] = Task(feature)
        guard_task: Task[None] = Task(self.check_input(content))
        tasks: Collection[Task[Result | None]] = (
            feature_task,
            guard_task,
        )
        completed, tasks = await wait(
            tasks,
            return_when=FIRST_COMPLETED,
        )

        if completed is guard_task:
            if exc := guard_task.exception():
                feature_task.cancel()
                raise exc

        await wait(
            tasks,
            return_when=ALL_COMPLETED,
        )

        if exc := guard_task.exception():
            raise exc

        return feature_task.result()

    @overload
    @classmethod
    async def input_guarded_stream[Result](
        cls,
        content: Multimodal,
        /,
        feature: Coroutine[None, None, AsyncIterable[Result]],
        **extra: Any,
    ) -> AsyncIterable[Result]: ...

    @overload
    async def input_guarded_stream[Result](
        self,
        content: Multimodal,
        /,
        feature: Coroutine[None, None, AsyncIterable[Result]],
        **extra: Any,
    ) -> AsyncIterable[Result]: ...

    @statemethod
    async def input_guarded_stream[Result](
        self,
        content: Multimodal,
        /,
        feature: Coroutine[None, None, AsyncIterable[Result]],
        **extra: Any,
    ) -> AsyncIterable[Result]:
        # TODO: implement input modetaion guardrail check in praralell during streaming
        return await feature

    @overload
    @classmethod
    async def check_output(
        cls,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> None: ...

    @overload
    async def check_output(
        self,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> None: ...

    @statemethod
    async def check_output(
        self,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> None:
        content = MultimodalContent.of(content)
        try:
            await self.output_checking(
                content,
                **extra,
            )

        except GuardrailsOutputModerationException:
            raise

        except GuardrailsModerationException as exc:
            raise GuardrailsOutputModerationException(
                f"Output moderation guardrails triggered: {exc}",
                content=content,
                violations=exc.violations,
                replacement=exc.replacement,
                meta=exc.meta,
            ) from exc

        except GuardrailsException as exc:
            raise GuardrailsOutputModerationException(
                f"Output moderation guardrails triggered: {exc}",
                content=content,
                violations={str(exc): 1.0},
                meta=exc.meta,
            ) from exc

        except Exception as exc:
            raise GuardrailsFailure(
                f"Output moderation guardrails failed: {exc}",
                cause=exc,
                meta={"error_type": exc.__class__.__name__},
            ) from exc

    input_checking: GuardrailsModerationChecking = _no_moderation
    output_checking: GuardrailsModerationChecking = _no_moderation
