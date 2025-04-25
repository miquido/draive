from collections.abc import Callable, Coroutine
from typing import Any

from haiway import State, ctx, mimic_function

from draive.guardrails.types import (
    GuardrailsContentException,
    GuardrailsContentVerifying,
    GuardrailsInputException,
    GuardrailsOutputException,
)
from draive.multimodal import Multimodal, MultimodalContent

__all__ = ("ContentGuardrails",)


async def _no_verification(
    content: MultimodalContent,
    /,
    **extra: Any,
) -> None:
    pass


class ContentGuardrails(State):
    @classmethod
    async def verify_input(
        cls,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> None:
        try:
            await ctx.state(cls).input_verifying(MultimodalContent.of(content))

        except GuardrailsContentException as exc:
            raise GuardrailsInputException(
                f"Input guardrails triggered: {exc}",
                content=exc.content,
                violations=exc.violations,
                replacement=exc.replacement,
            ) from exc

    @classmethod
    async def with_input_verification[Result](
        cls,
        function: Callable[[MultimodalContent], Coroutine[None, None, Result]],
        /,
        **extra: Any,
    ) -> Callable[[MultimodalContent], Coroutine[None, None, Result]]:
        async def verified(input: MultimodalContent) -> Result:  # noqa: A002
            await ctx.state(cls).input_verifying(input)
            return await function(input)

        return mimic_function(function, within=verified)

    @classmethod
    async def verify_output(
        cls,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> None:
        try:
            await ctx.state(cls).output_verifying(MultimodalContent.of(content))

        except GuardrailsContentException as exc:
            raise GuardrailsOutputException(
                f"Output guardrails triggered: {exc}",
                content=exc.content,
                violations=exc.violations,
                replacement=exc.replacement,
            ) from exc

    @classmethod
    async def with_output_verification[**Args](
        cls,
        function: Callable[Args, Coroutine[None, None, MultimodalContent]],
        /,
        **extra: Any,
    ) -> Callable[Args, Coroutine[None, None, MultimodalContent]]:
        async def verified(*args: Args.args, **kwargs: Args.kwargs) -> MultimodalContent:
            output: MultimodalContent = await function(*args, **kwargs)
            await ctx.state(cls).output_verifying(output)
            return output

        return mimic_function(function, within=verified)

    input_verifying: GuardrailsContentVerifying = _no_verification
    output_verifying: GuardrailsContentVerifying = _no_verification
