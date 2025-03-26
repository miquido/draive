from typing import Literal, overload

from cohere import AsyncClientV2
from haiway import getenv_str

__all__ = [
    "CohereAPI",
]


class CohereAPI:
    @overload
    def __init__(
        self,
        provider: Literal["bedrock"],
        /,
        *,
        timeout: float = 60.0,
    ) -> None: ...

    @overload
    def __init__(
        self,
        provider: Literal["cohere"] = "cohere",
        /,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None: ...

    def __init__(
        self,
        provider: Literal["cohere", "bedrock"] = "cohere",
        /,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._provider: Literal["cohere", "bedrock"] = provider
        self._base_url: str | None = base_url or getenv_str("COHERE_SERVER_URL")
        self._api_key: str | None = api_key or getenv_str("COHERE_API_KEY")
        self._timeout: float = timeout
        self._client: AsyncClientV2 = self._prepare_client()

    def _prepare_client(self) -> AsyncClientV2:
        match self._provider:
            case "cohere":
                return AsyncClientV2(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    timeout=self._timeout,
                )

            case "bedrock":
                from draive.cohere.bedrock import AsyncBedrockClientV2

                return AsyncBedrockClientV2(
                    aws_access_key=getenv_str("AWS_ACCESS_KEY_ID"),
                    aws_secret_key=getenv_str("AWS_ACCESS_KEY"),
                    aws_region=getenv_str("AWS_DEFAULT_REGION"),
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
