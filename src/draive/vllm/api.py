from collections.abc import Mapping
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from haiway import getenv_str
from openai import AsyncOpenAI, not_given

__all__ = ("VLLMAPI",)


@lru_cache(maxsize=8)
def _provider_label(
    base_url: str,
    /,
) -> str:
    # the label reaches observability backends, a base url can carry credentials
    # within its userinfo and a token within its query - only the host identifies
    # which deployment produced the data
    try:
        parts = urlsplit(base_url)
        return f"vllm@{urlunsplit((parts.scheme, parts.hostname or '', parts.path, '', ''))}"

    except Exception:
        return "vllm"


class VLLMAPI:
    __slots__ = (
        "_api_key",
        "_base_url",
        "_client",
        "_default_headers",
        "_extra",
        "_timeout",
    )

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        **extra: Any,
    ) -> None:
        self._base_url: str = (
            base_url or getenv_str("VLLM_BASE_URL") or getenv_str("OPENAI_BASE_URL", required=True)
        )
        self._api_key: str | None = api_key
        self._default_headers: Mapping[str, str] | None = default_headers
        self._extra: Mapping[str, Any] = extra
        self._timeout: float | None = timeout
        self._client: AsyncOpenAI  # initialized later

    @property
    def _provider(self) -> str:
        # derived instead of stored, it stays available to instances prepared
        # without going through the initializer
        return _provider_label(self._base_url)

    def _prepare_client(self) -> AsyncOpenAI:
        # a copy - the placeholders are only defaults, each can be provided explicitly
        extra: dict[str, Any] = {**self._extra}
        return AsyncOpenAI(
            # using placeholders to prevent reading environment variables targeting actual OpenAI
            # api_key included - vLLM endpoints require no credentials and an actual
            # OPENAI_API_KEY must never be sent to an arbitrary base_url
            api_key=self._api_key or getenv_str("VLLM_API_KEY", default="vllm"),
            organization=extra.pop("organization", None) or "vllm",
            project=extra.pop("project", None) or "vllm",
            base_url=self._base_url,
            default_headers=self._default_headers,
            # `None` means "no timeout", omit it to keep the SDK default instead
            timeout=self._timeout if self._timeout is not None else not_given,
            **extra,
        )

    async def _initialize_client(self) -> None:
        assert not hasattr(self, "_client")  # nosec: B101
        self._client = self._prepare_client()
        await self._client.__aenter__()

    async def _deinitialize_client(self) -> None:
        try:
            await self._client.__aexit__(
                None,
                None,
                None,
            )

        finally:
            del self._client
