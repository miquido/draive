from typing import Literal, cast

from anthropic import AsyncAnthropic, not_given

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
                    # `None` means "no timeout", omit it to keep the SDK default instead
                    timeout=self._timeout if self._timeout is not None else not_given,
                )

            case "bedrock":
                from anthropic.lib.bedrock import AsyncAnthropicBedrock

                return cast(
                    # Bedrock client does not inherit from AsyncAnthropic
                    # while providing all of the API we are using
                    AsyncAnthropic,
                    AsyncAnthropicBedrock(
                        base_url=self._base_url,
                        aws_region=self._aws_region,
                        max_retries=0,  # disable library retries
                        # `None` means "no timeout", omit it to keep the SDK default instead
                        timeout=self._timeout if self._timeout is not None else not_given,
                    ),
                )

    async def _initialize_client(self) -> None:
        assert not hasattr(self, "_client")  # nosec: B101
        self._client = self._prepare_client()
        await self._client.__aenter__()

    async def _deinitialize_client(self) -> None:
        # closing keeps the instance usable - deleting it would break streams
        # still held by a caller past the scope with an attribute error
        try:
            await self._client.__aexit__(
                None,
                None,
                None,
            )

        finally:
            del self._client
