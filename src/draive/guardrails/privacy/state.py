from typing import Any, overload

from haiway import State, statemethod

from draive.guardrails.privacy.types import (
    GuardrailsAnonymizedContent,
    GuardrailsContentAnonymizing,
)
from draive.multimodal import Multimodal, MultimodalContent

__all__ = ("GuardrailsAnonymization",)


class GuardrailsAnonymization(State):
    @overload
    @classmethod
    async def anonymize(
        cls,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> GuardrailsAnonymizedContent: ...

    @overload
    async def anonymize(
        self,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> GuardrailsAnonymizedContent: ...

    @statemethod
    async def anonymize(
        self,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> GuardrailsAnonymizedContent:
        return await self.anonymizing(
            MultimodalContent.of(content),
            **extra,
        )

    anonymizing: GuardrailsContentAnonymizing
