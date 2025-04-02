from collections.abc import Iterable, Set
from types import TracebackType
from typing import Literal, final

from haiway import State

from draive.openai.api import OpenAIAPI
from draive.openai.embedding import OpenAIEmbedding
from draive.openai.guardrails import OpenAIContentFiltering
from draive.openai.images import OpenAIImageGeneration
from draive.openai.lmm_generation import OpenAILMMGeneration
from draive.openai.tokenization import OpenAITokenization

__all__ = ("OpenAI",)


@final
class OpenAI(
    OpenAILMMGeneration,
    OpenAIEmbedding,
    OpenAIImageGeneration,
    OpenAIContentFiltering,
    OpenAITokenization,
    OpenAIAPI,
):
    """
    Access to OpenAI services, can be used to prepare various functionalities like lmm.
    """

    __slots__ = ("_disposable_state",)

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        azure_api_endpoint: str | None = None,
        azure_api_version: str | None = None,
        azure_deployment: str | None = None,
        disposable_state: Set[Literal["lmm", "text_embedding", "image_generation"]] | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            organization=organization,
            azure_api_endpoint=azure_api_endpoint,
            azure_api_version=azure_api_version,
            azure_deployment=azure_deployment,
        )

        self._disposable_state: frozenset[Literal["lmm", "text_embedding", "image_generation"]] = (
            frozenset(disposable_state)
            if disposable_state is not None
            else frozenset(("lmm", "text_embedding", "image_generation"))
        )

    async def __aenter__(self) -> Iterable[State]:
        await self._initialize_client()
        state: list[State] = []
        if "lmm" in self._disposable_state:
            state.append(self.lmm())

        if "text_embedding" in self._disposable_state:
            state.append(self.text_embedding())

        if "image_generation" in self._disposable_state:
            state.append(self.image_generation())

        return state

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._deinitialize_client()
