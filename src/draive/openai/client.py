from asyncio import gather
from collections.abc import Iterable
from os import getenv
from typing import Self, final

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai._types import NOT_GIVEN
from openai.types import Moderation, ModerationCreateResponse
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from draive.openai.config import OpenAIChatConfig, OpenAIEmbeddingConfig
from draive.scope import ScopeDependency

__all__ = [
    "OpenAIClient",
]


@final
class OpenAIClient(ScopeDependency):
    @classmethod
    def prepare(cls) -> Self:
        return cls(
            api_key=getenv("AZURE_API_KEY") or getenv("OPENAI_API_KEY"),
            organization=getenv("OPENAI_ORGANIZATION"),
            azure_api_endpoint=getenv("AZURE_API_BASE"),
            azure_api_version=getenv("AZURE_API_VERSION"),
            azure_deployment=getenv("AZURE_DEPLOYMENT_NAME"),
        )

    def __init__(  # noqa: PLR0913
        self,
        api_key: str | None,
        organization: str | None = None,
        azure_api_endpoint: str | None = None,
        azure_api_version: str | None = None,
        azure_deployment: str | None = None,
    ):
        # if all AZURE settings provided use it as provider
        if azure_api_endpoint and azure_deployment and azure_api_version:
            self._client: AsyncOpenAI = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_api_endpoint,
                azure_deployment=azure_deployment,
                api_version=azure_api_version,
                organization=organization,
            )
        # otherwise try using OpenAI
        else:
            self._client: AsyncOpenAI = AsyncOpenAI(
                api_key=api_key,
                organization=organization,
            )

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    async def chat_completion(
        self,
        config: OpenAIChatConfig,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam],
    ) -> ChatCompletion:
        return await self._client.chat.completions.create(
            messages=messages,
            model=config.model,
            frequency_penalty=config.frequency_penalty or NOT_GIVEN,
            max_tokens=config.max_tokens or NOT_GIVEN,
            n=1,
            response_format=config.response_format or NOT_GIVEN,
            seed=config.seed or NOT_GIVEN,
            stream=False,
            temperature=config.temperature,
            tools=tools or NOT_GIVEN,
            top_p=config.top_p or NOT_GIVEN,
            timeout=config.timeout or NOT_GIVEN,
        )

    async def embedding(
        self,
        config: OpenAIEmbeddingConfig,
        inputs: Iterable[str],
    ) -> list[list[float]]:
        inputs_list: list[str] = list(inputs)
        return [
            embeddings.embedding
            for response in await gather(
                *[
                    self._client.embeddings.create(
                        input=list(inputs_list[index : index + config.batch_size]),
                        model=config.model,
                        dimensions=config.dimensions or NOT_GIVEN,
                        encoding_format=config.encoding_format or NOT_GIVEN,
                        timeout=config.timeout or NOT_GIVEN,
                    )
                    for index in range(0, len(inputs_list), config.batch_size)
                ]
            )
            for embeddings in response.data
        ]

    async def moderation_check(
        self,
        text: str,
    ) -> Moderation:
        response: ModerationCreateResponse = await self._client.moderations.create(
            input=text,
        )
        return response.results[0]  # TODO: check API about multiple results

    async def close(self):
        await self._client.close()
