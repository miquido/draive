from random import uniform
from typing import Literal, Self, final, overload

from anthropic import AsyncAnthropic, AsyncStream
from anthropic import RateLimitError as AnthropicRateLimitError
from anthropic._types import NOT_GIVEN, NotGiven
from anthropic.types import Message, MessageParam, RawMessageStreamEvent, ToolParam
from anthropic.types.message_create_params import ToolChoice, ToolChoiceToolChoiceTool

from draive.anthropic.config import AnthropicConfig
from draive.scope import ScopeDependency
from draive.types import RateLimitError
from draive.utils import getenv_str, not_missing

__all__ = [
    "AnthropicClient",
]


@final
class AnthropicClient(ScopeDependency):
    @classmethod
    def prepare(cls) -> Self:
        return cls(
            api_key=getenv_str("ANTHROPIC_API_KEY"),
        )

    def __init__(
        self,
        api_key: str | None,
    ) -> None:
        self._client: AsyncAnthropic = AsyncAnthropic(
            api_key=api_key,
            max_retries=0,  # disable library retries
        )

    @property
    def client(self) -> AsyncAnthropic:
        return self._client

    @overload
    async def completion(
        self,
        *,
        config: AnthropicConfig,
        instruction: str,
        messages: list[MessageParam],
        tools: list[ToolParam] | None = None,
        tool_requirement: ToolChoiceToolChoiceTool | bool | None = False,
        stream: Literal[True],
    ) -> AsyncStream[RawMessageStreamEvent]: ...

    @overload
    async def completion(
        self,
        *,
        config: AnthropicConfig,
        instruction: str,
        messages: list[MessageParam],
        tools: list[ToolParam] | None = None,
        tool_requirement: ToolChoiceToolChoiceTool | bool | None = False,
    ) -> Message: ...

    async def completion(  # noqa: PLR0913
        self,
        *,
        config: AnthropicConfig,
        instruction: str,
        messages: list[MessageParam],
        tools: list[ToolParam] | None = None,
        tool_requirement: ToolChoiceToolChoiceTool | bool | None = False,
        stream: bool = False,
    ) -> AsyncStream[RawMessageStreamEvent] | Message:
        tool_choice: ToolChoice | NotGiven
        match tool_requirement:
            case None:
                tool_choice = NOT_GIVEN

            case False:
                tool_choice = {"type": "auto"} if tools else NOT_GIVEN

            case True:
                assert tools, "Can't require tools use without tools"  # nosec: B101
                tool_choice = {"type": "any"} if tools else NOT_GIVEN

            case tool:
                assert tools, "Can't require tools use without tools"  # nosec: B101
                tool_choice = tool

        try:
            return await self._client.messages.create(
                model=config.model,
                system=instruction,
                messages=messages,
                tools=tools or NOT_GIVEN,
                tool_choice=tool_choice,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p if not_missing(config.top_p) else NOT_GIVEN,
                timeout=config.timeout if not_missing(config.timeout) else NOT_GIVEN,
                stream=stream,
            )

        except AnthropicRateLimitError as exc:  # retry on rate limit after delay
            if delay := exc.response.headers.get("Retry-After"):
                raise RateLimitError(
                    retry_after=delay + uniform(0.0, 0.3)  # nosec: B311 # add small random delay
                ) from exc

            else:
                raise exc

    async def dispose(self) -> None:
        await self._client.close()
