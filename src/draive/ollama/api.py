from typing import Any

from haiway import getenv_str
from ollama import AsyncClient

__all__ = ("OllamaAPI",)


class OllamaAPI:
    __slots__ = (
        "_client",
        "_server_url",
        "_timeout",
    )

    def __init__(
        self,
        server_url: str | None = None,
        timeout: float | None = None,
        **extra: Any,
    ) -> None:
        # Prefer explicit server_url, fallback to env vars
        self._server_url: str | None = (
            server_url or getenv_str("OLLAMA_SERVER_URL") or getenv_str("OLLAMA_HOST")
        )
        self._timeout: float | None = timeout
        self._client: AsyncClient  # initialized later

    def _prepare_client(self) -> AsyncClient:
        return AsyncClient(
            host=self._server_url,
            timeout=self._timeout,
        )

    async def _initialize_client(self) -> None:
        assert not hasattr(self, "_client")  # nosec: B101
        self._client = self._prepare_client()
        await self._client.__aenter__()

    async def _deinitialize_client(self) -> None:
        try:
            await self._client.__aexit__(  # pyright: ignore[reportUnknownMemberType]
                None,
                None,
                None,
            )

        finally:
            del self._client
