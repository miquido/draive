from collections.abc import Collection, Iterable
from types import TracebackType
from typing import Literal, final

from haiway import State

from draive.openai.api import OpenAIAPI
from draive.openai.embedding import OpenAIEmbedding
from draive.openai.guardrails import OpenAIContentModereation
from draive.openai.images import OpenAIImageGeneration
from draive.openai.lmm_generation import OpenAILMMGeneration
from draive.openai.lmm_session import OpenAIRealtimeLMM
from draive.openai.tokenization import OpenAITokenization

__all__ = ("OpenAI",)


type Features = Literal[
    "lmm",
    "lmm_session",
    "text_embedding",
    "image_generation",
    "content_guardrails",
]


@final
class OpenAI(
    OpenAILMMGeneration,
    OpenAIRealtimeLMM,
    OpenAIEmbedding,
    OpenAIImageGeneration,
    OpenAIContentModereation,
    OpenAITokenization,
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
        features: Collection[Features] | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            organization=organization,
            azure_api_endpoint=azure_api_endpoint,
            azure_api_version=azure_api_version,
            azure_deployment=azure_deployment,
        )

        self._features: frozenset[Features] = (
            frozenset(features)
            if features is not None
            else frozenset(
                (
                    "lmm",
                    "lmm_session",
                    "text_embedding",
                    "image_generation",
                    "content_guardrails",
                )
            )
        )

    async def __aenter__(self) -> Iterable[State]:
        await self._initialize_client()
        state: list[State] = []
        if "lmm" in self._features:
            state.append(self.lmm())

        if "lmm_session" in self._features:
            state.append(self.lmm_session())

        if "text_embedding" in self._features:
            state.append(self.text_embedding())

        if "image_generation" in self._features:
            state.append(self.image_generation())

        if "content_guardrails" in self._features:
            state.append(self.content_guardrails())

        return state

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._deinitialize_client()
