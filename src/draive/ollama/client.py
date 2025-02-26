from types import TracebackType
from typing import final

from draive.ollama.api import OllamaAPI
from draive.ollama.embedding import OllamaEmbedding
from draive.ollama.lmm_invoking import OllamaLMMInvoking

__all__ = [
    "Ollama",
]


@final
class Ollama(
    OllamaLMMInvoking,
    OllamaEmbedding,
    OllamaAPI,
):
    """
    Access to Ollama services, can be used to prepare various functionalities like lmm.
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
