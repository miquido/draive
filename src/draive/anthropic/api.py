from typing import Literal, cast

from anthropic import AsyncAnthropic

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
        timeout: float | None = None,
    ) -> None:
        self._provider: Literal["anthropic", "bedrock"] = provider
        self._base_url: str | None = base_url
        self._api_key: str | None = api_key
        self._aws_region: str | None = aws_region
        self._timeout: float | None = timeout
        self._client: AsyncAnthropic  # lazily initialized

    def _prepare_client(self) -> AsyncAnthropic:
        match self._provider:
            case "anthropic":
                return AsyncAnthropic(
                    base_url=self._base_url,
                    api_key=self._api_key,
                    max_retries=0,  # disable library retries
                    timeout=self._timeout,
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
