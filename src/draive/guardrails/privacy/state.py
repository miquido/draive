from typing import Any

from haiway import State, ctx

from draive.guardrails.privacy.types import (
    GuardrailsAnonymizedContent,
    GuardrailsContentAnonymizing,
)
from draive.multimodal import Multimodal, MultimodalContent

__all__ = ("GuardrailsAnonymization",)


class GuardrailsAnonymization(State):
    @classmethod
    async def anonymize(
        cls,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> GuardrailsAnonymizedContent:
        return await ctx.state(cls).anonymizing(MultimodalContent.of(content))

    anonymizing: GuardrailsContentAnonymizing
