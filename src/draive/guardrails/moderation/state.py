from typing import Any

from haiway import State, ctx

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
    @classmethod
    async def check_input(
        cls,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> None:
        try:
            await ctx.state(cls).input_checking(MultimodalContent.of(content))

        except GuardrailsModerationException as exc:
            raise GuardrailsInputModerationException(
                f"Input moderation guardrails triggered: {exc}",
                content=exc.content,
                violations=exc.violations,
                replacement=exc.replacement,
            ) from exc

    @classmethod
    async def check_output(
        cls,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> None:
        try:
            await ctx.state(cls).output_checking(MultimodalContent.of(content))

        except GuardrailsModerationException as exc:
            raise GuardrailsOutputModerationException(
                f"Output moderation guardrails triggered: {exc}",
                content=exc.content,
                violations=exc.violations,
                replacement=exc.replacement,
            ) from exc

    input_checking: GuardrailsModerationChecking = _no_moderation
    output_checking: GuardrailsModerationChecking = _no_moderation
