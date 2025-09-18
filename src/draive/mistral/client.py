from collections.abc import Collection, Iterable
from types import TracebackType
from typing import Any, final

from haiway import State

from draive.embedding import TextEmbedding
from draive.guardrails import GuardrailsModeration
from draive.mistral.api import MistralAPI
from draive.mistral.completions import MistralCompletions
from draive.mistral.embedding import MistralEmbedding
from draive.mistral.moderation import MistralContentModeration
from draive.models import GenerativeModel

__all__ = ("Mistral",)


@final
class Mistral(
    MistralCompletions,
    MistralEmbedding,
    MistralContentModeration,
    MistralAPI,
):
    """
    Access to Mistral services, can be used to prepare various functionalities like lmm.
    """

    __slots__ = ("_features",)

    def __init__(
        self,
        server_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        features: Collection[type[GenerativeModel | TextEmbedding | GuardrailsModeration]]
        | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(
            server_url=server_url,
            api_key=api_key,
            timeout=timeout,
            **extra,
        )

        self._features: frozenset[type[State]] = (
            frozenset(features)
            if features is not None
            else frozenset((GenerativeModel, TextEmbedding, GuardrailsModeration))
        )

    async def __aenter__(self) -> Iterable[State]:
        await self._initialize_client()
        state: list[State] = []

        if GenerativeModel in self._features:
            state.append(self.generative_model())

        if TextEmbedding in self._features:
            state.append(self.text_embedding())

        if GuardrailsModeration in self._features:
            state.append(self.content_guardrails())

        return state

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._deinitialize_client()
