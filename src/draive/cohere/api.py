from cohere import AsyncClient
from haiway import getenv_str

__all__ = [
    "CohereAPI",
]


class CohereAPI:
    def __init__(
        self,
        server_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._server_url: str | None = server_url or getenv_str("COHERE_SERVER_URL")
        self._api_key: str | None = api_key or getenv_str("COHERE_API_KEY")
        self._timeout: float = timeout
        self._client = self._prepare_client()

    def _prepare_client(self) -> AsyncClient:
        return AsyncClient(
            api_key=self._api_key,
            base_url=self._server_url,
            timeout=self._timeout,
        )

    async def _initialize_client(self) -> None:
        await self._client._client_wrapper.httpx_client.httpx_client.aclose()
        self._client = self._prepare_client()
        await self._client.__aenter__()

    async def _deinitialize_client(self) -> None:
        await self._client.__aexit__(
            None,
            None,
            None,
        )
