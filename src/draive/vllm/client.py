from types import TracebackType
from typing import final

from draive.vllm.api import VLLMAPI
from draive.vllm.embedding import VLLMEmbedding
from draive.vllm.lmm_invoking import VLLMLMMInvoking
from draive.vllm.lmm_streaming import VLLMLMMStreaming

__all__ = [
    "VLLM",
]


@final
class VLLM(
    VLLMLMMInvoking,
    VLLMLMMStreaming,
    VLLMEmbedding,
    VLLMAPI,
):
    """
    Access to VLLM services, can be used to prepare various functionalities like lmm.
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
