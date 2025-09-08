from collections.abc import Collection, Iterable
from types import TracebackType
from typing import final

from haiway import State

from draive.bedrock.api import BedrockAPI
from draive.bedrock.converse import BedrockConverse
from draive.bedrock.guardrails import BedrockGuardrails
from draive.guardrails import GuardrailsModeration
from draive.models import GenerativeModel

__all__ = ("Bedrock",)


@final
class Bedrock(
    BedrockConverse,
    BedrockGuardrails,
    BedrockAPI,
):
    __slots__ = ("_features",)

    def __init__(
        self,
        *,
        features: Collection[type[GenerativeModel | GuardrailsModeration]] | None = None,
        aws_region: str | None = None,
    ) -> None:
        super().__init__(aws_region=aws_region)
        self._features: frozenset[type[State]] = (
            frozenset(features)
            if features is not None
            else frozenset((GenerativeModel, GuardrailsModeration))
        )

    async def __aenter__(self) -> Iterable[State]:
        await self._initialize_client()
        state: list[State] = []
        if GenerativeModel in self._features:
            state.append(self.generative_model())

        if GuardrailsModeration in self._features:
            state.append(self.guardrails_moderation())

        return state

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._deinitialize_client()
