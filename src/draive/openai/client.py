from asyncio import gather, sleep
from collections.abc import Iterable
from itertools import chain
from random import uniform
from typing import Literal, Self, final, overload

from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream, RateLimitError
from openai._types import NOT_GIVEN
from openai.types import Moderation, ModerationCreateResponse
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolParam,
)
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.image import Image
from openai.types.images_response import ImagesResponse

from draive.helpers import getenv_str
from draive.openai.config import (
    OpenAIChatConfig,
    OpenAIEmbeddingConfig,
    OpenAIImageGenerationConfig,
)
from draive.scope import ScopeDependency

__all__ = [
    "OpenAIClient",
]


@final
class OpenAIClient(ScopeDependency):
    @classmethod
    def prepare(cls) -> Self:
        return cls(
            api_key=getenv_str("AZURE_OPENAI_API_KEY") or getenv_str("OPENAI_API_KEY"),
            organization=getenv_str("OPENAI_ORGANIZATION"),
            azure_api_endpoint=getenv_str("AZURE_OPENAI_API_BASE"),
            azure_api_version=getenv_str("AZURE_OPENAI_API_VERSION"),
            azure_deployment=getenv_str("AZURE_OPENAI_DEPLOYMENT_NAME"),
        )

    def __init__(  # noqa: PLR0913
        self,
        api_key: str | None,
        organization: str | None = None,
        azure_api_endpoint: str | None = None,
        azure_api_version: str | None = None,
        azure_deployment: str | None = None,
    ) -> None:
        # if all AZURE settings provided use it as provider
        if azure_api_endpoint and azure_deployment and azure_api_version:
            self._client: AsyncOpenAI = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_api_endpoint,
                azure_deployment=azure_deployment,
                api_version=azure_api_version,
                organization=organization,
            )
        # otherwise try using OpenAI default
        else:
            self._client: AsyncOpenAI = AsyncOpenAI(
                api_key=api_key,
                organization=organization,
            )

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @overload
    async def chat_completion(
        self,
        *,
        config: OpenAIChatConfig,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam],
        stream: Literal[True],
    ) -> AsyncStream[ChatCompletionChunk]:
        ...

    @overload
    async def chat_completion(
        self,
        *,
        config: OpenAIChatConfig,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam],
        suggested_tool: ChatCompletionNamedToolChoiceParam | None,
        stream: Literal[True],
    ) -> AsyncStream[ChatCompletionChunk]:
        ...

    @overload
    async def chat_completion(
        self,
        *,
        config: OpenAIChatConfig,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam],
        suggested_tool: ChatCompletionNamedToolChoiceParam | None = None,
    ) -> ChatCompletion:
        ...

    async def chat_completion(  # noqa: PLR0913
        self,
        *,
        config: OpenAIChatConfig,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam],
        suggested_tool: ChatCompletionNamedToolChoiceParam | None = None,
        stream: bool = False,
    ) -> AsyncStream[ChatCompletionChunk] | ChatCompletion:
        return await self._client.chat.completions.create(
            messages=messages,
            model=config.model,
            frequency_penalty=config.frequency_penalty or NOT_GIVEN,
            max_tokens=config.max_tokens or NOT_GIVEN,
            n=1,
            response_format=config.response_format or NOT_GIVEN,
            seed=config.seed or NOT_GIVEN,
            stream=stream,
            temperature=config.temperature,
            tools=tools or NOT_GIVEN,
            tool_choice=(suggested_tool or "auto") if tools else NOT_GIVEN,
            top_p=config.top_p or NOT_GIVEN,
            timeout=config.timeout or NOT_GIVEN,
        )

    async def embedding(
        self,
        config: OpenAIEmbeddingConfig,
        inputs: Iterable[str],
    ) -> list[list[float]]:
        inputs_list: list[str] = list(inputs)
        return list(
            chain(
                *await gather(
                    *[
                        self._create_text_embedding(
                            texts=list(inputs_list[index : index + config.batch_size]),
                            model=config.model,
                            dimensions=config.dimensions,
                            encoding_format=config.encoding_format,
                            timeout=config.timeout,
                        )
                        for index in range(0, len(inputs_list), config.batch_size)
                    ]
                )
            )
        )

    async def _create_text_embedding(  # noqa: PLR0913
        self,
        texts: list[str],
        model: str,
        dimensions: int | None,
        encoding_format: Literal["float", "base64"] | None,
        timeout: float | None,
    ) -> list[list[float]]:
        try:
            response: CreateEmbeddingResponse = await self._client.embeddings.create(
                input=texts,
                model=model,
                dimensions=dimensions or NOT_GIVEN,
                encoding_format=encoding_format or NOT_GIVEN,
                timeout=timeout or NOT_GIVEN,
            )
            return [element.embedding for element in response.data]

        except RateLimitError:  # always retry on rate limit
            # wait between 0.1s and 1s before next attempt
            await sleep(delay=uniform(0.1, 1))  # nosec: B311
            return await self._create_text_embedding(
                texts=texts,
                model=model,
                dimensions=dimensions,
                encoding_format=encoding_format,
                timeout=timeout,
            )

    async def moderation_check(
        self,
        text: str,
    ) -> Moderation:
        response: ModerationCreateResponse = await self._client.moderations.create(
            input=text,
        )
        return response.results[0]  # TODO: check API about multiple results

    async def generate_image(
        self,
        config: OpenAIImageGenerationConfig,
        instruction: str,
    ) -> Image:
        response: ImagesResponse = await self._client.images.generate(
            model=config.model,
            n=1,
            prompt=instruction,
            quality=config.quality,
            size=config.size,
            style=config.style,
            timeout=config.timeout,
            response_format=config.response_format,
        )
        return response.data[0]

    async def dispose(self):
        await self._client.close()
