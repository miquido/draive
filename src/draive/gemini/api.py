from collections.abc import Mapping
from typing import Any

from google.genai import Client
from haiway import getenv_str

__all__ = [
    "GeminiAPI",
]


class GeminiAPI:
    def __init__(
        self,
        api_key: str | None = None,
        **extra: Any,
    ) -> None:
        self._api_key: str | None = api_key or getenv_str("GEMINI_API_KEY")
        self._extra: Mapping[str, Any] = extra
        self._client = Client(
            api_key=self._api_key,
            **self._extra,
        )
