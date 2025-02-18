from types import TracebackType
from typing import final

from draive.mistral.api import MistralAPI
from draive.mistral.embedding import MistralEmbedding
from draive.mistral.lmm_invoking import MistralLMMInvoking
from draive.mistral.lmm_streaming import MistralLMMStreaming
from draive.mistral.tokenization import MistralTokenization

__all__ = [
    "Mistral",
]


@final
class Mistral(
    MistralLMMInvoking,
    MistralLMMStreaming,
    MistralEmbedding,
    MistralTokenization,
    MistralAPI,
):
    """
    Access to Mistral services, can be used to prepare various functionalities like lmm.
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
