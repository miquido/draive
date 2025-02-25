from types import TracebackType
from typing import final

from draive.cohere.api import CohereAPI
from draive.cohere.embedding import CohereEmbedding

__all__ = [
    "Cohere",
]


@final
class Cohere(
    CohereEmbedding,
    CohereAPI,
):
    """
    Access to Cohere services, can be used to prepare various functionalities like embedding.
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
