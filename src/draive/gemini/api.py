from typing import Any

from google.genai import Client
from google.genai.client import HttpOptionsDict  # pyright: ignore[reportPrivateImportUsage]
from haiway import getenv_str

__all__ = ("GeminiAPI",)


class GeminiAPI:
    __slots__ = ("_client",)

    def __init__(
        self,
        api_key: str | None,
        vertexai: bool | None,
        http_options: HttpOptionsDict | None,
        **extra: Any,
    ) -> None:
        self._client = Client(
            api_key=api_key or getenv_str("GEMINI_API_KEY"),
            vertexai=vertexai,
            http_options=http_options,
            **extra,
        )
