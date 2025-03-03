from types import TracebackType
from typing import final

from draive.anthropic.api import AnthropicAPI
from draive.anthropic.lmm_invoking import AnthropicLMMInvoking

__all__ = [
    "Anthropic",
]


@final
class Anthropic(
    AnthropicLMMInvoking,
    AnthropicAPI,
):
    async def __aenter__(self) -> None:
        await self._initialize_client()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._deinitialize_client()
