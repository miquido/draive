from typing import Literal

from cohere import AsyncClientV2
from haiway import getenv_str

from draive.cohere.bedrock import CohereBedrock

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
        timeout: float | None = None,
    ) -> None:
        self._provider: Literal["cohere", "bedrock"] = provider
        self._timeout: float | None = timeout
        self._base_url: str | None = base_url or getenv_str("COHERE_SERVER_URL")
        self._api_key: str | None = api_key or getenv_str("COHERE_API_KEY")
        self._aws_region: str | None = aws_region or getenv_str("AWS_BEDROCK_REGION")
        self._client: AsyncClientV2 | CohereBedrock  # initialized later

    def _prepare_client(self) -> AsyncClientV2 | CohereBedrock:
        match self._provider:
            case "cohere":
                return AsyncClientV2(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    timeout=self._timeout,
                )

            case "bedrock":
                return CohereBedrock(aws_region=self._aws_region)
