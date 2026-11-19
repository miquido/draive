from collections.abc import Collection, Iterable, MutableSequence, Set
from types import TracebackType
from typing import Any, final

from google.genai.types import HttpOptionsDict
from haiway import State

from draive.embedding import TextEmbedding
from draive.gemini.api import GeminiAPI
from draive.gemini.decisions import GeminiDecisions
from draive.gemini.embedding import GeminiEmbedding
from draive.gemini.generating import GeminiGenerating
from draive.gemini.live import GeminiLive
from draive.generation import DecisionGeneration
from draive.models import GenerativeModel, RealtimeGenerativeModel

__all__ = ("Gemini",)


@final
class Gemini(
    GeminiLive,
    GeminiGenerating,
    GeminiEmbedding,
    GeminiDecisions,
    GeminiAPI,
):
    """
    Access to Gemini services, can be used to prepare various functionalities like lmm.
    """

    __slots__ = (
        "_default_features",
        "_features",
    )

    def __init__(
        self,
        api_key: str | None = None,
        vertexai: bool | None = None,
        http_options: HttpOptionsDict | None = None,
        features: Collection[
            type[GenerativeModel | RealtimeGenerativeModel | TextEmbedding | DecisionGeneration]
        ]
        | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            vertexai=vertexai,
            http_options=http_options,
            **extra,
        )

        self._default_features: bool = features is None
        self._features: Set[type[State]] = (
            frozenset(features)
            if features is not None
            else frozenset(
                (GenerativeModel, RealtimeGenerativeModel, TextEmbedding, DecisionGeneration)
            )
        )

    async def __aenter__(self) -> Iterable[State]:
        await self._initialize_client()
        state: MutableSequence[State] = []
        if GenerativeModel in self._features:
            state.append(GenerativeModel(generating=self.completion))

        if RealtimeGenerativeModel in self._features:
            state.append(RealtimeGenerativeModel(session_preparing=self.session_prepare))

        if TextEmbedding in self._features:
            state.append(TextEmbedding(embedding=self.create_texts_embedding))

        # decisions are not available through Vertex AI, keep the model based ones by default
        if DecisionGeneration in self._features and not (
            self._default_features and self._client.vertexai
        ):
            state.append(DecisionGeneration(generating=self.decide))

        return state

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._deinitialize_client()
