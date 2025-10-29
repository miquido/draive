from typing import Literal

from cohere import AsyncClientV2
from haiway import getenv_str

__all__ = ("CohereAPI",)


class CohereAPI:
    __slots__ = (
        "_api_key",
        "_aws_region",
        "_base_url",
        "_client",
        "_provider",
        "_timeout",
    )

    def __init__(
        self,
        provider: Literal["cohere", "bedrock"] = "cohere",
        /,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        aws_region: str | None = None,
    ) -> None:
        self._provider: Literal["cohere", "bedrock"] = provider
        self._base_url: str | None = base_url or getenv_str("COHERE_SERVER_URL")
        self._api_key: str | None = api_key or getenv_str("COHERE_API_KEY")
        self._aws_region: str | None = aws_region or getenv_str("AWS_BEDROCK_REGION")

        self._client: AsyncClientV2 = self._prepare_client()

    def _prepare_client(self) -> AsyncClientV2:
        match self._provider:
            case "cohere":
                return AsyncClientV2(
                    api_key=self._api_key,
                    base_url=self._base_url,
                )

            case "bedrock":
                from draive.cohere.bedrock import AsyncBedrockClientV2

                return AsyncBedrockClientV2(aws_region=self._aws_region)

    async def _initialize_client(self) -> None:
        await self._deinitialize_client()

        self._client = self._prepare_client()
        await self._client.__aenter__()  # pyright: ignore[reportUnknownMemberType]

    async def _deinitialize_client(self) -> None:
        await self._client.__aexit__(None, None, None)  # pyright: ignore[reportUnknownMemberType]
