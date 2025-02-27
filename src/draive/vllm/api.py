from collections.abc import Mapping
from typing import Any

from haiway import getenv_str
from openai import AsyncOpenAI

__all__ = [
    "VLLMAPI",
]


class VLLMAPI:
    def __init__(
        self,
        base_url: str | None = None,
        **extra: Any,
    ) -> None:
        self._base_url: str | None = base_url or getenv_str("VLLM_BASE_URL")
        self._extra: Mapping[str, Any] = extra
        self._client: AsyncOpenAI = self._prepare_client()

    def _prepare_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            # using placeholders to prevent reading environment variables targeting actual OpenAI
            organization="vllm",
            project="vllm",
            base_url=self._base_url,
            **self._extra,
        )

    async def _initialize_client(self) -> None:
        await self._client.close()
        self._client = self._prepare_client()
        await self._client.__aenter__()

    async def _deinitialize_client(self) -> None:
        await self._client.__aexit__(
            None,
            None,
            None,
        )
