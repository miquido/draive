from collections.abc import AsyncIterable
from typing import Literal, Self, cast, final, overload

from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessage,
)

from draive.helpers import getenv_str
from draive.mistral.config import MistralChatConfig
from draive.scope import ScopeDependency

__all__ = [
    "MistralClient",
]


@final
class MistralClient(ScopeDependency):
    @classmethod
    def prepare(cls) -> Self:
        return cls(
            api_key=getenv_str("MISTRAL_API_KEY"),
            endpoint=getenv_str("MISTRAL_ENDPOINT"),
        )

    def __init__(
        self,
        api_key: str | None,
        endpoint: str | None = None,
    ) -> None:
        if endpoint:
            self._client: MistralAsyncClient = MistralAsyncClient(
                api_key=api_key,
                endpoint=endpoint,
            )
        else:
            self._client: MistralAsyncClient = MistralAsyncClient(
                api_key=api_key,
            )

    @overload
    async def chat_completion(
        self,
        *,
        config: MistralChatConfig,
        messages: list[ChatMessage],
        tools: list[dict[str, object]],
        stream: Literal[True],
    ) -> AsyncIterable[ChatCompletionStreamResponse]:
        ...

    @overload
    async def chat_completion(
        self,
        *,
        config: MistralChatConfig,
        messages: list[ChatMessage],
        tools: list[dict[str, object]],
        suggest_tools: bool,
        stream: Literal[True],
    ) -> AsyncIterable[ChatCompletionStreamResponse]:
        ...

    @overload
    async def chat_completion(
        self,
        *,
        config: MistralChatConfig,
        messages: list[ChatMessage],
        tools: list[dict[str, object]],
        suggest_tools: bool = False,
    ) -> ChatCompletionResponse:
        ...

    async def chat_completion(  # noqa: PLR0913
        self,
        *,
        config: MistralChatConfig,
        messages: list[ChatMessage],
        tools: list[dict[str, object]],
        suggest_tools: bool = False,
        stream: bool = False,
    ) -> AsyncIterable[ChatCompletionStreamResponse] | ChatCompletionResponse:
        if stream:
            return self._client.chat_stream(
                messages=messages,
                model=config.model,
                max_tokens=config.max_tokens,
                response_format=cast(dict[str, str], config.response_format),
                random_seed=config.seed,
                temperature=config.temperature,
                tools=tools,
                top_p=config.top_p,
            )
        else:
            return await self._client.chat(
                messages=messages,
                model=config.model,
                max_tokens=config.max_tokens,
                response_format=cast(dict[str, str], config.response_format),
                random_seed=config.seed,
                temperature=config.temperature,
                tools=tools,
                tool_choice=("any" if suggest_tools else "auto") if tools else None,
                top_p=config.top_p,
            )

    async def dispose(self) -> None:
        await self._client.close()
