from collections.abc import Mapping
from typing import Any

from google.genai import Client
from google.genai.types import HttpOptionsDict
from haiway import getenv_bool, getenv_str

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
            http_options=self._resolved_http_options(),
            **self._extra,
        )

    def _resolved_http_options(self) -> HttpOptionsDict | None:
        # the developer api rejects system instructions within v1, the sdk default,
        # use v1beta unless a version or vertex ai was selected explicitly
        if (
            self._vertexai
            or getenv_bool("GOOGLE_GENAI_USE_VERTEXAI", default=False)
            or getenv_bool("GOOGLE_GENAI_USE_ENTERPRISE", default=False)
            or getenv_str("GOOGLE_GENAI_API_VERSION")
            or (self._http_options is not None and "api_version" in self._http_options)
        ):
            return self._http_options

        options: HttpOptionsDict = (
            self._http_options.copy() if self._http_options is not None else HttpOptionsDict()
        )
        options["api_version"] = "v1beta"
        return options

    async def _deinitialize_client(self) -> None:
        try:
            await self._client.aio.aclose()
            self._client.close()

        finally:
            del self._client
