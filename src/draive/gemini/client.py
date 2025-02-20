from types import TracebackType
from typing import final

from draive.gemini.api import GeminiAPI
from draive.gemini.embedding import GeminiEmbedding
from draive.gemini.lmm_invoking import GeminiLMMInvoking
from draive.gemini.tokenization import GeminiTokenization

__all__ = [
    "Gemini",
]


@final
class Gemini(
    GeminiLMMInvoking,
    GeminiEmbedding,
    GeminiTokenization,
    GeminiAPI,
):
    """
    Access to Gemini services, can be used to prepare various functionalities like lmm.
    """

    async def __aenter__(self) -> None:
        pass

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass
