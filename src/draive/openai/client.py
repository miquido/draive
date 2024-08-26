from asyncio import gather
from collections.abc import Sequence
from itertools import chain
from random import uniform
from typing import Literal, Self, cast, final, overload

from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream
from openai import RateLimitError as OpenAIRateLimitError
from openai._types import NOT_GIVEN, NotGiven
from openai.types import Moderation, ModerationCreateResponse
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.image import Image
from openai.types.images_response import ImagesResponse

from draive.openai.config import (
    OpenAIChatConfig,
    OpenAIEmbeddingConfig,
    OpenAIImageGenerationConfig,
)
from draive.scope import ScopeDependency
from draive.types import RateLimitError
from draive.utils import freeze, getenv_str, not_missing

__all__ = [
    "OpenAIClient",
]


@final
class OpenAIClient(ScopeDependency):
    @classmethod
    def prepare(cls) -> Self:
        return cls(
            base_url=getenv_str("OPENAI_BASE_URL"),
            api_key=getenv_str("AZURE_OPENAI_API_KEY") or getenv_str("OPENAI_API_KEY"),
            organization=getenv_str("OPENAI_ORGANIZATION"),
            azure_api_endpoint=getenv_str("AZURE_OPENAI_API_BASE"),
            azure_api_version=getenv_str("AZURE_OPENAI_API_VERSION"),
            azure_deployment=getenv_str("AZURE_OPENAI_DEPLOYMENT_NAME"),
        )

    def __init__(  # noqa: PLR0913
        self,
        base_url: str | None,
        api_key: str | None,
        organization: str | None = None,
        azure_api_endpoint: str | None = None,
        azure_api_version: str | None = None,
        azure_deployment: str | None = None,
    ) -> None:
        # if all AZURE settings were provided use it as provider
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
                base_url=base_url,
                api_key=api_key,
                organization=organization,
            )

        freeze(self)

    @overload
    async def chat_completion(
        self,
        *,
        config: OpenAIChatConfig,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        stream: Literal[True],
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def chat_completion(
        self,
        *,
        config: OpenAIChatConfig,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
    ) -> ChatCompletion: ...

    async def chat_completion(
        self,
        *,
        config: OpenAIChatConfig,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        stream: bool = False,
    ) -> AsyncStream[ChatCompletionChunk] | ChatCompletion:
        try:
            return await self._client.chat.completions.create(
                messages=messages,
                model=config.model,
                frequency_penalty=config.frequency_penalty
                if not_missing(config.frequency_penalty)
                else NOT_GIVEN,
                max_tokens=config.max_tokens if not_missing(config.max_tokens) else NOT_GIVEN,
                n=1,
                response_format=cast(ResponseFormat, config.response_format)
                if not_missing(config.response_format)
                else NOT_GIVEN,
                seed=config.seed if not_missing(config.seed) else NOT_GIVEN,
                stream=stream,
                temperature=config.temperature,
                tools=tools or NOT_GIVEN,
                tool_choice=tool_choice if tools else NOT_GIVEN,
                parallel_tool_calls=True if tools else NOT_GIVEN,
                top_p=config.top_p if not_missing(config.top_p) else NOT_GIVEN,
                timeout=config.timeout if not_missing(config.timeout) else NOT_GIVEN,
                stream_options={"include_usage": True} if stream else NOT_GIVEN,
                stop=config.stop_sequences if not_missing(config.stop_sequences) else NOT_GIVEN,
            )

        except OpenAIRateLimitError as exc:  # retry on rate limit after delay
            if delay := exc.response.headers.get("Retry-After"):
                try:
                    raise RateLimitError(
                        retry_after=float(delay) + uniform(0.0, 0.3)  # nosec: B311 # add small random delay
                    ) from exc

                except ValueError:
                    raise exc from None

            else:
                raise exc

    async def embedding(
        self,
        config: OpenAIEmbeddingConfig,
        inputs: Sequence[str],
    ) -> list[list[float]]:
        return list(
            chain(
                *await gather(
                    *[
                        self._create_text_embedding(
                            texts=inputs[index : index + config.batch_size],
                            model=config.model,
                            dimensions=config.dimensions
                            if not_missing(config.dimensions)
                            else NOT_GIVEN,
                            encoding_format=cast(Literal["float", "base64"], config.encoding_format)
                            if not_missing(config.encoding_format)
                            else NOT_GIVEN,
                            timeout=config.timeout if not_missing(config.timeout) else NOT_GIVEN,
                        )
                        for index in range(0, len(inputs), config.batch_size)
                    ]
                )
            )
        )

    async def _create_text_embedding(
        self,
        texts: Sequence[str],
        model: str,
        dimensions: int | NotGiven,
        encoding_format: Literal["float", "base64"] | NotGiven,
        timeout: float | NotGiven,
    ) -> list[list[float]]:
        while True:
            try:
                response: CreateEmbeddingResponse = await self._client.embeddings.create(
                    input=list(texts),
                    model=model,
                    dimensions=dimensions,
                    encoding_format=encoding_format,
                    timeout=timeout,
                )
                return [element.embedding for element in response.data]

            except OpenAIRateLimitError as exc:  # always retry on rate limit after delay
                if delay := exc.response.headers.get("Retry-After"):
                    try:
                        raise RateLimitError(
                            retry_after=float(delay) + uniform(0.0, 0.3)  # nosec: B311 # add small random delay
                        ) from exc

                    except ValueError:
                        raise exc from None

                else:
                    raise exc

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
            timeout=config.timeout if not_missing(config.timeout) else NOT_GIVEN,
            response_format=config.response_format,
        )
        return response.data[0]

    async def dispose(self) -> None:
        await self._client.close()
