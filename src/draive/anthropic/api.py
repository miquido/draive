from anthropic import AsyncAnthropic
from haiway import getenv_str

__all__ = [
    "AnthropicAPI",
]


class AnthropicAPI:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url: str | None = base_url
        self._api_key: str | None = api_key or getenv_str("ANTHROPIC_API_KEY")
        self._timeout: float = timeout
        self._client: AsyncAnthropic = self._prepare_client()

    def _prepare_client(self) -> AsyncAnthropic:
        return AsyncAnthropic(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            max_retries=0,  # disable library retries
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
