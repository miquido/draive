from typing import Any, overload

from haiway import State, statemethod

from draive.guardrails.moderation.types import (
    GuardrailsInputModerationException,
    GuardrailsModerationChecking,
    GuardrailsModerationException,
    GuardrailsOutputModerationException,
)
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
        try:
            await self.input_checking(
                MultimodalContent.of(content),
                **extra,
            )

        except GuardrailsModerationException as exc:
            raise GuardrailsInputModerationException(
                f"Input moderation guardrails triggered: {exc}",
                content=exc.content,
                violations=exc.violations,
                replacement=exc.replacement,
            ) from exc

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
        try:
            await self.output_checking(
                MultimodalContent.of(content),
                **extra,
            )

        except GuardrailsModerationException as exc:
            raise GuardrailsOutputModerationException(
                f"Output moderation guardrails triggered: {exc}",
                content=exc.content,
                violations=exc.violations,
                replacement=exc.replacement,
            ) from exc

    input_checking: GuardrailsModerationChecking = _no_moderation
    output_checking: GuardrailsModerationChecking = _no_moderation
