from typing import Any

from haiway import getenv_str
from mistralai.client import Mistral as MistralClient

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
        self._client: MistralClient  # initialized lazily

    def _prepare_client(self) -> MistralClient:
        return MistralClient(
            api_key=self._api_key,
            server_url=self._server_url,
            timeout_ms=int(self._timeout * 1000.0) if self._timeout is not None else None,
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
            # `Mistral` implicitly creates a synchronous httpx client as well, which only
            # its synchronous context manager exit closes - without it the client leaks
            self._client.__exit__(  # pyright: ignore[reportUnknownMemberType]
                None,
                None,
                None,
            )

        finally:
            del self._client
