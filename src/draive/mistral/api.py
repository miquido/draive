from typing import Any

from haiway import getenv_str
from mistralai import Mistral as MistralClient

__all__ = ("MistralAPI",)


class MistralAPI:
    __slots__ = (
        "_api_key",
        "_client",
        "_server_url",
        "_timeout",
    )

    def __init__(
        self,
        server_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        **extra: Any,
    ) -> None:
        self._server_url: str = server_url or getenv_str(
            "MISTRAL_SERVER_URL",
            default="https://api.mistral.ai",
        )
        self._api_key: str | None = api_key or getenv_str("MISTRAL_API_KEY")
        self._timeout: float | None = timeout
        self._client: MistralClient = self._prepare_client()

    def _prepare_client(self) -> MistralClient:
        return MistralClient(
            api_key=self._api_key,
            server_url=self._server_url,
            timeout_ms=int(self._timeout * 1000.0) if self._timeout is not None else None,
        )

    async def _initialize_client(self) -> None:
        if self._client.sdk_configuration.async_client is not None:
            await self._client.sdk_configuration.async_client.aclose()

        self._client = self._prepare_client()
        await self._client.__aenter__()

    async def _deinitialize_client(self) -> None:
        await self._client.__aexit__(  # pyright: ignore[reportUnknownMemberType]
            None,
            None,
            None,
        )
