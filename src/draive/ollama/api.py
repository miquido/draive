from haiway import getenv_str
from ollama import AsyncClient

__all__ = ("OllamaAPI",)


class OllamaAPI:
    __slots__ = (
        "_client",
        "_server_url",
    )

    def __init__(
        self,
        server_url: str | None = None,
    ) -> None:
        self._server_url: str | None = server_url or getenv_str("OLLAMA_SERVER_URL")
        self._client = self._prepare_client()

    def _prepare_client(self) -> AsyncClient:
        return AsyncClient(host=self._server_url)

    async def _initialize_client(self) -> None:
        await self._client._client.aclose()
        self._client = self._prepare_client()
        await self._client._client.__aenter__()

    async def _deinitialize_client(self) -> None:
        await self._client._client.__aexit__(
            None,
            None,
            None,
        )
