from typing import Any

from haiway import getenv_str
from openai import AsyncAzureOpenAI, AsyncOpenAI, not_given

__all__ = ("OpenAIAPI",)


class OpenAIAPI:
    __slots__ = (
        "_api_key",
        "_azure_api_endpoint",
        "_azure_api_version",
        "_azure_deployment",
        "_base_url",
        "_client",
        "_organization",
        "_timeout",
    )

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        azure_api_endpoint: str | None = None,
        azure_api_version: str | None = None,
        azure_deployment: str | None = None,
        timeout: float | None = None,
        **extra: Any,
    ) -> None:
        self._base_url: str | None = base_url or getenv_str("OPENAI_BASE_URL")
        self._api_key: str | None = (
            api_key or getenv_str("AZURE_OPENAI_API_KEY") or getenv_str("OPENAI_API_KEY")
        )
        self._organization: str | None = organization or getenv_str("OPENAI_ORGANIZATION")
        self._azure_api_endpoint: str | None = azure_api_endpoint or getenv_str(
            "AZURE_OPENAI_API_BASE"
        )
        self._azure_deployment: str | None = azure_deployment or getenv_str(
            "AZURE_OPENAI_DEPLOYMENT_NAME"
        )
        self._azure_api_version: str | None = azure_api_version or getenv_str(
            "AZURE_OPENAI_API_VERSION"
        )
        self._timeout: float | None = timeout
        self._client: AsyncOpenAI  # initialized later

    def _prepare_client(self) -> AsyncOpenAI:
        if self._azure_api_endpoint:
            return AsyncAzureOpenAI(
                # `None` allows the SDK to use its own `AZURE_OPENAI_AD_TOKEN` fallbacks,
                # an empty string would slip past those checks and fail afterwards
                api_key=self._api_key,
                azure_endpoint=self._azure_api_endpoint,
                azure_deployment=self._azure_deployment,
                api_version=self._azure_api_version,
                organization=self._organization,
                # `None` means "no timeout", omit it to keep the SDK default instead
                timeout=self._timeout if self._timeout is not None else not_given,
            )

        else:  # otherwise try using OpenAI
            return AsyncOpenAI(
                base_url=self._base_url,
                # OpenAI compatible servers behind a custom base_url may require no credentials,
                # the SDK requires a non empty key anyways - use a placeholder for those.
                # Without a custom base_url let the SDK raise its own error instead.
                api_key=self._api_key or ("openai" if self._base_url else None),
                organization=self._organization,
                timeout=self._timeout if self._timeout is not None else not_given,
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
