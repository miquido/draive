from asyncio import gather
from collections.abc import AsyncIterable, Iterable
from itertools import chain
from typing import Literal, Self, cast, final, overload

from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from mistralai.models.embeddings import EmbeddingResponse

from draive.helpers import getenv_str, when_missing
from draive.mistral.config import MistralChatConfig, MistralEmbeddingConfig
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
    ) -> AsyncIterable[ChatCompletionStreamResponse]: ...

    @overload
    async def chat_completion(
        self,
        *,
        config: MistralChatConfig,
        messages: list[ChatMessage],
        tools: list[dict[str, object]],
        suggest_tools: bool,
        stream: Literal[True],
    ) -> AsyncIterable[ChatCompletionStreamResponse]: ...

    @overload
    async def chat_completion(
        self,
        *,
        config: MistralChatConfig,
        messages: list[ChatMessage],
        tools: list[dict[str, object]],
        suggest_tools: bool = False,
    ) -> ChatCompletionResponse: ...

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
                max_tokens=when_missing(config.max_tokens, default=None),
                response_format=cast(dict[str, str], config.response_format),
                random_seed=when_missing(config.seed, default=None),
                temperature=config.temperature,
                tools=tools,
                top_p=when_missing(config.top_p, default=None),
            )
        else:
            return await self._client.chat(
                messages=messages,
                model=config.model,
                max_tokens=when_missing(config.max_tokens, default=None),
                response_format=cast(dict[str, str], config.response_format),
                random_seed=when_missing(config.seed, default=None),
                temperature=config.temperature,
                tools=tools,
                tool_choice=("any" if suggest_tools else "auto") if tools else None,
                top_p=when_missing(config.top_p, default=None),
            )

    async def embedding(
        self,
        config: MistralEmbeddingConfig,
        inputs: Iterable[str],
    ) -> list[list[float]]:
        inputs_list: list[str] = list(inputs)
        return list(
            chain(
                *await gather(
                    *[
                        self._create_text_embedding(
                            model=config.model,
                            texts=list(inputs_list[index : index + config.batch_size]),
                        )
                        for index in range(0, len(inputs_list), config.batch_size)
                    ]
                )
            )
        )

    async def _create_text_embedding(
        self,
        model: str,
        texts: list[str],
    ) -> list[list[float]]:
        response: EmbeddingResponse = await self._client.embeddings(
            model=model,
            input=texts,
        )
        return [element.embedding for element in response.data]

    async def dispose(self) -> None:
        await self._client.close()
