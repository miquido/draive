from types import TracebackType
from typing import Any, ClassVar, Self, final

from haiway import asynchronous, getenv_str, not_missing

from draive.bedrock.config import BedrockChatConfig
from draive.bedrock.models import ChatCompletionResponse, ChatMessage, ChatTool

__all__ = [
    "BedrockClient",
]


@final
class BedrockClient:
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
        region_name: str | None = None,
        access_key_id: str | None = None,
        access_key: str | None = None,
    ) -> None:
        self._region_name: str | None = region_name or getenv_str("AWS_DEFAULT_REGION")
        self._access_key_id: str | None = access_key_id or getenv_str("AWS_ACCESS_KEY_ID")
        self._access_key: str | None = access_key or getenv_str("AWS_ACCESS_KEY")
        self._client: Any

    # preparing it lazily on demand, boto does a lot of stuff on initialization
    @asynchronous
    def initialize(self) -> None:
        if hasattr(self, "_client"):
            return  # already initialized

        import boto3  # pyright: ignore[reportMissingTypeStubs]

        self._client = boto3.Session(  # pyright: ignore[reportUnknownMemberType]
            aws_access_key_id=self._access_key_id,
            aws_secret_access_key=self._access_key,
            region_name=self._region_name,
        ).client("bedrock-runtime")

    async def chat_completion(
        self,
        *,
        config: BedrockChatConfig,
        instruction: str | None,
        messages: list[ChatMessage],
        tools: list[ChatTool],
        require_tool: str | bool,
    ) -> ChatCompletionResponse:
        return await self._create_chat_completion(
            model=config.model,
            temperature=config.temperature,
            top_p=config.top_p if not_missing(config.top_p) else None,
            max_tokens=config.max_tokens if not_missing(config.max_tokens) else None,
            instruction=instruction,
            messages=messages,
            tools=tools,
            require_tool=require_tool,
            stop=config.stop_sequences if not_missing(config.stop_sequences) else None,
        )

    @asynchronous
    def _create_chat_completion(  # noqa: PLR0913
        self,
        model: str,
        temperature: float,
        top_p: float | None,
        max_tokens: int | None,
        instruction: str | None,
        messages: list[ChatMessage],
        tools: list[ChatTool],
        require_tool: str | bool,
        stop: list[str] | None,
    ) -> ChatCompletionResponse:
        parameters: dict[str, Any] = {
            "modelId": model,
            "messages": messages,
            "inferenceConfig": {
                "temperature": temperature,
            },
        }

        if instruction:
            parameters["system"] = [
                {
                    "text": instruction,
                }
            ]

        if tools:
            toolChoice: Any
            match require_tool:
                case False:
                    toolChoice = {"auto": {}}

                case True:
                    toolChoice = {"any": {}}

                case tool:
                    toolChoice = {"tool": {"name": tool}}

            parameters["toolConfig"] = {
                "tools": [{"toolSpec": tool} for tool in tools],
                "toolChoice": toolChoice,
            }

        if max_tokens:
            parameters["inferenceConfig"]["maxTokens"] = max_tokens
        if top_p is not None:
            parameters["inferenceConfig"]["topP"] = top_p
        if stop:
            parameters["inferenceConfig"]["stopSequences"] = stop

        return self._client.converse(**parameters)

    async def __aenter__(self) -> None:
        await self.initialize()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass
