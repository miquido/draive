from typing import Literal, cast

from anthropic import AsyncAnthropic
from haiway import getenv_str

__all__ = ("AnthropicAPI",)


class AnthropicAPI:
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
        provider: Literal["anthropic", "bedrock"] = "anthropic",
        /,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        aws_region: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._provider: Literal["anthropic", "bedrock"] = provider
        self._base_url: str | None = base_url
        self._api_key: str | None = api_key or getenv_str("ANTHROPIC_API_KEY")
        self._aws_region: str | None = aws_region or getenv_str("AWS_BEDROCK_REGION")
        self._timeout: float = timeout
        self._client: AsyncAnthropic = self._prepare_client()

    def _prepare_client(self) -> AsyncAnthropic:
        match self._provider:
            case "anthropic":
                return AsyncAnthropic(
                    base_url=self._base_url,
                    api_key=self._api_key,
                    timeout=self._timeout,
                    max_retries=0,  # disable library retries
                )

            case "bedrock":
                from anthropic import AsyncAnthropicBedrock

                return cast(
                    AsyncAnthropic,  # it is not actually the same but for our purpose it works
                    AsyncAnthropicBedrock(
                        base_url=self._base_url,
                        aws_region=self._aws_region,
                        timeout=self._timeout,
                    ),
                )

    async def _initialize_client(self) -> None:
        await self._client._client.aclose()
        self._client = self._prepare_client()
        await self._client.__aenter__()

    async def _deinitialize_client(self) -> None:
        await self._client.__aexit__(
            None,
            None,
            None,
        )
