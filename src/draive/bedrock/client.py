from collections.abc import Collection, Iterable
from types import TracebackType
from typing import Literal, final

from haiway import State

from draive.bedrock.api import BedrockAPI
from draive.bedrock.guardrails import BedrockGuardrais
from draive.bedrock.lmm_generation import BedrockLMMGeneration

__all__ = ("Bedrock",)

type Features = Literal[
    "lmm",
    "content_guardrails",
]


@final
class Bedrock(
    BedrockLMMGeneration,
    BedrockGuardrais,
    BedrockAPI,
):
    __slots__ = ("_features",)

    def __init__(
        self,
        *,
        features: Collection[Features] | None = None,
        aws_region: str | None = None,
    ) -> None:
        super().__init__(aws_region=aws_region)
        self._features: frozenset[Features] = (
            frozenset(features)
            if features is not None
            else frozenset(
                (
                    "lmm",
                    "content_guardrails",
                )
            )
        )

    async def __aenter__(self) -> Iterable[State]:
        await self._initialize_client()
        state: list[State] = []
        if "lmm" in self._features:
            state.append(self.lmm())

        if "content_guardrails" in self._features:
            state.append(self.content_guardrails())

        return state

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._deinitialize_client()
