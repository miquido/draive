from random import uniform
from types import TracebackType
from typing import ClassVar, Literal, Self, final, overload

from anthropic import AsyncAnthropic, AsyncStream
from anthropic import RateLimitError as AnthropicRateLimitError
from anthropic._types import NOT_GIVEN, NotGiven
from anthropic.types import Message, MessageParam, RawMessageStreamEvent, ToolParam
from anthropic.types.message_create_params import ToolChoice, ToolChoiceToolChoiceTool
from haiway import getenv_str, not_missing

from draive.anthropic.config import AnthropicConfig
from draive.utils import RateLimitError

__all__ = [
    "AnthropicClient",
]


@final
class AnthropicClient:
    _SHARED: ClassVar[Self]

    @classmethod
    def shared(cls) -> Self:
        if shared := getattr(cls, "_SHARED", None):
            return shared

        else:
            cls._SHARED = cls()  # pyright: ignore[reportConstantRedefinition]
            return cls._SHARED

    def __init__(
        self,
        api_key: str | None = None,
    ) -> None:
        self._api_key: str | None = api_key or getenv_str("ANTHROPIC_API_KEY")
        self._client: AsyncAnthropic = self._initialize_client()

    def _initialize_client(self) -> AsyncAnthropic:
        return AsyncAnthropic(
            api_key=self._api_key,
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
        instruction: str | None,
        messages: list[MessageParam],
        tools: list[ToolParam] | None = None,
        tool_choice: ToolChoiceToolChoiceTool | Literal["auto", "any", "none"] = "auto",
        stream: Literal[True],
    ) -> AsyncStream[RawMessageStreamEvent]: ...

    @overload
    async def completion(
        self,
        *,
        config: AnthropicConfig,
        instruction: str | None,
        messages: list[MessageParam],
        tools: list[ToolParam] | None = None,
        tool_choice: ToolChoiceToolChoiceTool | Literal["auto", "any", "none"] = "auto",
    ) -> Message: ...

    async def completion(  # noqa: PLR0913
        self,
        *,
        config: AnthropicConfig,
        instruction: str | None,
        messages: list[MessageParam],
        tools: list[ToolParam] | None = None,
        tool_choice: ToolChoiceToolChoiceTool | Literal["auto", "any", "none"] = "auto",
        stream: bool = False,
    ) -> AsyncStream[RawMessageStreamEvent] | Message:
        selected_tool_choice: ToolChoice | NotGiven
        match tool_choice:
            case "auto":
                selected_tool_choice = {"type": "auto"} if tools else NOT_GIVEN

            case "none":
                selected_tool_choice = NOT_GIVEN

            case "any":
                assert tools, "Can't require tools use without tools"  # nosec: B101
                selected_tool_choice = {"type": "any"} if tools else NOT_GIVEN

            case tool:
                assert tools, "Can't require tools use without tools"  # nosec: B101
                selected_tool_choice = tool

        try:
            return await self._client.messages.create(
                model=config.model,
                system=instruction or NOT_GIVEN,
                messages=messages,
                tools=tools or NOT_GIVEN,
                tool_choice=selected_tool_choice,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p if not_missing(config.top_p) else NOT_GIVEN,
                timeout=config.timeout if not_missing(config.timeout) else NOT_GIVEN,
                stop_sequences=config.stop_sequences
                if not_missing(config.stop_sequences)
                else NOT_GIVEN,
                stream=stream,
            )

        except AnthropicRateLimitError as exc:  # retry on rate limit after delay
            if delay := exc.response.headers.get("Retry-After"):
                try:
                    raise RateLimitError(
                        retry_after=float(delay) + uniform(0.0, 0.3)  # nosec: B311 # add small random delay
                    ) from exc

                except ValueError:
                    raise exc from None

            else:
                raise exc

    async def __aenter__(self) -> None:
        if self._client.is_closed():
            self._client = self._initialize_client()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._client.close()
