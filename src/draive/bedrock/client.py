from typing import Any, Self, final

from haiway import asynchronous, getenv_str, not_missing

from draive.bedrock.config import BedrockChatConfig
from draive.bedrock.models import ChatCompletionResponse, ChatMessage, ChatTool
from draive.scope import ScopeDependency  # pyright: ignore[reportDeprecated]

__all__ = [
    "BedrockClient",
]


@final
class BedrockClient(ScopeDependency):  # pyright: ignore[reportDeprecated]
    @classmethod
    def prepare(cls) -> Self:
        return cls(
            region_name=getenv_str("AWS_DEFAULT_REGION"),
            access_key_id=getenv_str("AWS_ACCESS_KEY_ID"),
            access_key=getenv_str("AWS_ACCESS_KEY"),
        )

    def __init__(
        self,
        region_name: str | None,
        access_key_id: str | None = None,
        access_key: str | None = None,
    ) -> None:
        self._region_name: str | None = region_name
        self._access_key_id: str | None = access_key_id
        self._access_key: str | None = access_key
        self._client: Any

    async def _initialize(self) -> None:
        if not hasattr(self, "_client"):
            await self._prepare_client()

    # preparing it lazily on demand, boto does a lot of stuff on initialization
    @asynchronous
    def _prepare_client(self) -> None:
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
        instruction: str,
        messages: list[ChatMessage],
        tools: list[ChatTool],
        require_tool: str | bool,
    ) -> ChatCompletionResponse:
        await self._initialize()
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
        instruction: str,
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

    async def dispose(self) -> None:
        pass
