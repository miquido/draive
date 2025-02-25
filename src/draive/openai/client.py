from types import TracebackType
from typing import final

from draive.openai.api import OpenAIAPI
from draive.openai.embedding import OpenAIEmbedding
from draive.openai.guardrails import OpenAIContentFiltering
from draive.openai.images import OpenAIImageGeneration
from draive.openai.lmm_invoking import OpenAILMMInvoking
from draive.openai.lmm_streaming import OpenAILMMStreaming
from draive.openai.tokenization import OpenAITokenization

__all__ = [
    "OpenAI",
]


@final
class OpenAI(
    OpenAILMMInvoking,
    OpenAILMMStreaming,
    OpenAIEmbedding,
    OpenAIImageGeneration,
    OpenAIContentFiltering,
    OpenAITokenization,
    OpenAIAPI,
):
    """
    Access to OpenAI services, can be used to prepare various functionalities like lmm.
    """

    async def __aenter__(self) -> None:
        await self._initialize_client()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._deinitialize_client()
