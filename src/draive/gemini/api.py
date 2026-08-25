from collections.abc import Mapping
from typing import Any

from google.genai import Client
from google.genai.types import HttpOptionsDict
from haiway import getenv_str

__all__ = ("GeminiAPI",)


class GeminiAPI:
    __slots__ = (
        "_api_key",
        "_client",
        "_extra",
        "_http_options",
        "_vertexai",
    )

    def __init__(
        self,
        api_key: str | None,
        vertexai: bool | None,
        http_options: HttpOptionsDict | None,
        **extra: Any,
    ) -> None:
        self._api_key: str | None = api_key or getenv_str("GEMINI_API_KEY")
        self._vertexai: bool | None = vertexai
        self._http_options: HttpOptionsDict | None = http_options
        self._extra: Mapping[str, Any] = extra
        self._client: Client  # lazily initialized

    async def _initialize_client(self) -> None:
        assert not hasattr(self, "_client")  # nosec: B101
        self._client = Client(
            api_key=self._api_key,
            vertexai=self._vertexai,
            http_options=self._http_options,
            **self._extra,
        )

    async def _deinitialize_client(self) -> None:
        try:
            await self._client.aio.aclose()
            self._client.close()

        finally:
            del self._client
