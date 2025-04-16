from typing import Any

from haiway import State, ctx

from draive.guardrails.types import GuardrailsContentVerifying
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
    async def verify(
        cls,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> None:
        await ctx.state(cls).verifying(MultimodalContent.of(content))

    verifying: GuardrailsContentVerifying = _no_verification
