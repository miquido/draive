from collections.abc import Mapping
from typing import Any, Literal

from haiway import HTTPClient, getenv_str
from haiway.httpx import HTTPXClient

__all__ = (
    "SystemOneAPI",
    "SystemOneImages",
)

SystemOneImages = Literal["assets", "images", "inline"]
"""How images are delivered - as request ``assets`` (TypeSafe), a request ``images`` list (Ollama)
or ``inline`` within the state (OpenJev)."""


class SystemOneAPI:
    __slots__ = (
        "_api_key",
        "_client",
        "_extra",
        "_headers",
        "_http",
        "_images",
        "_server_url",
        "_timeout",
    )

    def __init__(
        self,
        server_url: str | None,
        api_key: str | None,
        images: SystemOneImages,
        headers: Mapping[str, str] | None,
        timeout: float,
        **extra: Any,
    ) -> None:
        self._server_url: str = (
            server_url or getenv_str("SYSTEMONE_SERVER_URL") or "https://api.typesafe.ai"
        )
        self._api_key: str | None = (
            api_key or getenv_str("SYSTEMONE_API_KEY") or getenv_str("TYPESAFE_API_KEY")
        )
        self._images: SystemOneImages = images
        self._headers: Mapping[str, str] = headers or {}
        self._timeout: float = timeout
        self._extra: dict[str, Any] = extra
        self._client: HTTPXClient  # lazily initialized
        self._http: HTTPClient  # lazily initialized

    async def _initialize_client(self) -> None:
        assert not hasattr(self, "_client")  # nosec: B101
        self._client = HTTPXClient(
            base_url=self._server_url,
            headers={
                **self._headers,
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}),
            },
            timeout=self._timeout,
            **self._extra,
        )
        self._http = await self._client.__aenter__()

    async def _deinitialize_client(self) -> None:
        try:
            await self._client.__aexit__(None, None, None)

        finally:
            del self._http
            del self._client
