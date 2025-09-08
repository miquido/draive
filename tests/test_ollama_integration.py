import sys
import types
from collections.abc import AsyncGenerator

import pytest

# Create a fake 'ollama' module to avoid real dependency
ollama = types.ModuleType("ollama")


class DummyMessage:
    def __init__(
        self,
        *,
        role: str = "assistant",
        content: str | None = None,
        images: list[object] | None = None,
        tool_calls: list[object] | None = None,
        tool_name: str | None = None,
    ) -> None:
        self.role = role
        self.content = content
        self.images = images
        self.tool_calls = tool_calls
        self.tool_name = tool_name

    class ToolCall:
        class Function:
            def __init__(self, *, name: str, arguments: object) -> None:
                self.name = name
                self.arguments = arguments

        def __init__(self, *, function: "DummyMessage.ToolCall.Function") -> None:
            self.function = function


class DummyTool:
    class Function:
        def __init__(self, *, name: str, description: str | None, parameters: object) -> None:
            self.name = name
            self.description = description
            self.parameters = parameters

    def __init__(self, *, type: str, function: "DummyTool.Function") -> None:  # noqa: A002
        self.type = type
        self.function = function


class DummyImage:
    def __init__(self, *, value: object) -> None:
        self.value = value


class DummyOptions:
    def __init__(self, **_: object) -> None:  # ignore
        pass


class DummyChatResponse:
    def __init__(self, message: DummyMessage) -> None:
        self.message = message


class DummyAsyncClient:
    def __init__(self, *, host: str | None = None) -> None:
        self.host = host
        # Simulate underlying httpx.AsyncClient for lifecycle calls used by implementation
        self._client = self

    # httpx.AsyncClient-like stubs for __aenter__/__aexit__/aclose
    async def __aenter__(self):  # type: ignore[override]
        return self

    async def __aexit__(self, *_: object) -> None:  # type: ignore[override]
        return None

    async def aclose(self) -> None:
        return None

    # capture last call for assertions
    last_messages: list[DummyMessage] | None = None
    last_format: object | None = None

    async def chat(
        self,
        *,
        model: str,
        messages: list[DummyMessage],
        format: object | None,  # noqa: A002
        tools: list[DummyTool] | None,
        options: DummyOptions,
        stream: bool,
    ) -> DummyChatResponse | AsyncGenerator:
        # store arguments for assertions
        DummyAsyncClient.last_messages = messages
        DummyAsyncClient.last_format = format
        # For stream=False return a single response
        if not stream:
            # Return the last assistant message with a tool call and content
            msg = DummyMessage(
                role="assistant",
                content="Hello world",
                tool_calls=[
                    DummyMessage.ToolCall(
                        function=DummyMessage.ToolCall.Function(
                            name="ping",
                            arguments={"x": 1},
                        )
                    )
                ],
            )
            return DummyChatResponse(message=msg)

        # For stream=True return an async generator yielding cumulative content and final tool_calls
        async def gen():
            yield types.SimpleNamespace(message=DummyMessage(content="He"))
            yield types.SimpleNamespace(message=DummyMessage(content="Hel"))
            yield types.SimpleNamespace(message=DummyMessage(content="Hell"))
            yield types.SimpleNamespace(message=DummyMessage(content="Hello"))
            yield types.SimpleNamespace(
                message=DummyMessage(
                    tool_calls=[
                        DummyMessage.ToolCall(
                            function=DummyMessage.ToolCall.Function(name="pong", arguments={"y": 2})
                        )
                    ]
                )
            )

        return gen()


# Install fake module symbols expected by implementation
ollama.AsyncClient = DummyAsyncClient  # type: ignore[attr-defined]
ollama.ChatResponse = DummyChatResponse  # type: ignore[attr-defined]
ollama.EmbedResponse = type("EmbedResponse", (), {})  # minimal placeholder
ollama.Message = DummyMessage  # type: ignore[attr-defined]
ollama.Tool = DummyTool  # type: ignore[attr-defined]
ollama.Image = DummyImage  # type: ignore[attr-defined]
ollama.Options = DummyOptions  # type: ignore[attr-defined]
sys.modules.setdefault("ollama", ollama)


from draive.models import (  # noqa: E402
    ModelInput,
    ModelToolRequest,
    ModelToolResponse,
    ModelToolsDeclaration,
)
from draive.multimodal import MultimodalContent, TextContent  # noqa: E402
from draive.ollama.chat import OllamaChat  # noqa: E402
from draive.ollama.config import OllamaChatConfig  # noqa: E402


@pytest.mark.asyncio
async def test_ollama_chat_non_streaming_with_tools() -> None:
    chat = OllamaChat()

    config = OllamaChatConfig(model="dummy")
    context = (ModelInput.of(MultimodalContent.of("Hi")),)
    tools = ModelToolsDeclaration.of(
        {
            "name": "ping",
            "description": "",
            "parameters": None,
            "additionalProperties": False,
        }
    )

    out = await chat.completion(
        instructions="",
        context=context,
        tools=tools,
        output="text",
        stream=False,
        config=config,
    )

    # Produces text + tool call blocks
    blocks = list(out.blocks)
    assert any(isinstance(b, MultimodalContent) and b.to_str() == "Hello world" for b in blocks)
    assert any(isinstance(b, ModelToolRequest) and b.tool == "ping" for b in blocks)


@pytest.mark.asyncio
async def test_ollama_chat_streaming_emits_deltas_and_tool_request() -> None:
    chat = OllamaChat()

    config = OllamaChatConfig(model="dummy")
    context = (ModelInput.of(MultimodalContent.of("Hi")),)
    tools = ModelToolsDeclaration.of(
        {
            "name": "pong",
            "description": "",
            "parameters": None,
            "additionalProperties": False,
        }
    )

    chunks = []
    async for piece in chat.completion(
        instructions="",
        context=context,
        tools=tools,
        output="text",
        stream=True,
        config=config,
    ):
        chunks.append(piece)

    # Expect incremental text deltas: e.g., e, l, l, o (from He -> Hel -> Hell -> Hello)
    deltas = [c.text for c in chunks if isinstance(c, TextContent)]
    assert deltas and (deltas[-1] == "o" or deltas[-1].endswith("Hello world"))

    # Final tool request at the end
    tool_reqs = [c for c in chunks if isinstance(c, ModelToolRequest)]
    assert tool_reqs and tool_reqs[-1].tool in ("pong", "ping")


@pytest.mark.asyncio
async def test_ollama_chat_tool_response_is_encoded_with_tool_role() -> None:
    chat = OllamaChat()
    config = OllamaChatConfig(model="dummy")
    # Provide a tool response from user to ensure it becomes role=tool
    context = (
        ModelInput.of(
            ModelToolResponse.of(
                "id-1",
                tool="weather",
                content=MultimodalContent.of("sunny"),
            )
        ),
    )
    tools = ModelToolsDeclaration.none

    _ = await chat.completion(
        instructions="",
        context=context,
        tools=tools,
        output="text",
        stream=False,
        config=config,
    )

    # The last_messages should include a tool role message
    assert DummyAsyncClient.last_messages is not None
    assert any(
        m.role == "tool" and m.tool_name == "weather" for m in DummyAsyncClient.last_messages
    )


@pytest.mark.asyncio
async def test_ollama_chat_json_output_sets_format_json() -> None:
    chat = OllamaChat()
    config = OllamaChatConfig(model="dummy")
    context = (ModelInput.of(MultimodalContent.of("Hi")),)
    tools = ModelToolsDeclaration.none

    _ = await chat.completion(
        instructions="",
        context=context,
        tools=tools,
        output="json",
        stream=False,
        config=config,
    )

    assert DummyAsyncClient.last_format == "json"
