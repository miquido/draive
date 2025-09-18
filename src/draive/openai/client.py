from collections.abc import Collection, Iterable
from types import TracebackType
from typing import Any, final

from haiway import State

from draive.embedding import TextEmbedding
from draive.generation import ImageGeneration
from draive.guardrails import GuardrailsModeration
from draive.models import GenerativeModel, RealtimeGenerativeModel
from draive.openai.api import OpenAIAPI
from draive.openai.embedding import OpenAIEmbedding
from draive.openai.images import OpenAIImageGeneration
from draive.openai.moderation import OpenAIContentModeration
from draive.openai.realtime import OpenAIRealtime
from draive.openai.responses import OpenAIResponses

__all__ = ("OpenAI",)


@final
class OpenAI(
    OpenAIResponses,
    OpenAIRealtime,
    OpenAIEmbedding,
    OpenAIImageGeneration,
    OpenAIContentModeration,
    OpenAIAPI,
):
    """
    Access to OpenAI services, can be used to prepare various functionalities like lmm.
    """

    __slots__ = ("_features",)

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        azure_api_endpoint: str | None = None,
        azure_api_version: str | None = None,
        azure_deployment: str | None = None,
        timeout: float | None = None,
        features: Collection[type[State]] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            organization=organization,
            azure_api_endpoint=azure_api_endpoint,
            azure_api_version=azure_api_version,
            azure_deployment=azure_deployment,
            timeout=timeout,
            **extra,
        )

        self._features: frozenset[type[State]] = (
            frozenset(features)
            if features is not None
            else frozenset(
                (
                    GenerativeModel,
                    RealtimeGenerativeModel,
                    TextEmbedding,
                    ImageGeneration,
                    GuardrailsModeration,
                )
            )
        )

    async def __aenter__(self) -> Iterable[State]:
        await self._initialize_client()
        state: list[State] = []
        if GenerativeModel in self._features:
            state.append(self.generative_model())

        if RealtimeGenerativeModel in self._features:
            state.append(self.realtime_generative_model())

        if TextEmbedding in self._features:
            state.append(self.text_embedding())

        if ImageGeneration in self._features:
            state.append(self.image_generation())

        if GuardrailsModeration in self._features:
            state.append(self.moderation_guardrails())

        return state

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._deinitialize_client()
