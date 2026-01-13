from types import SimpleNamespace
from typing import Any

import pytest
from haiway import ctx

from draive.models import ModelOutputFailed, ModelOutputInvalid, ModelTools
from draive.ollama.chat import OllamaChat, _context_messages
from draive.ollama.config import OllamaChatConfig


def test_context_messages_includes_system_instructions() -> None:
    messages = list(_context_messages(instructions="Stay concise.", context=()))
    assert len(messages) == 1
    assert messages[0].role == "system"
    assert messages[0].content == "Stay concise."


@pytest.mark.asyncio
async def test_completion_translates_provider_errors_to_model_output_failed() -> None:
    async def _chat(**_: Any) -> Any:
        raise RuntimeError("connection lost")

    model = object.__new__(OllamaChat)
    model._client = SimpleNamespace(chat=_chat)

    async with ctx.scope("test"):
        with pytest.raises(ModelOutputFailed, match="connection lost"):
            await model._completion(
                instructions="system",
                tools=ModelTools.none,
                context=(),
                output="text",
                config=OllamaChatConfig(model="ollama-test"),
                prefill=None,
            )


@pytest.mark.asyncio
async def test_completion_translates_tool_argument_decode_error_to_model_output_invalid() -> None:
    async def _chat(**_: Any) -> Any:
        return SimpleNamespace(
            message=SimpleNamespace(
                content="",
                tool_calls=[
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name="lookup",
                            arguments='{"value": ',
                        )
                    )
                ],
            )
        )

    model = object.__new__(OllamaChat)
    model._client = SimpleNamespace(chat=_chat)

    async with ctx.scope("test"):
        with pytest.raises(ModelOutputInvalid, match="Tool arguments decoding error"):
            await model._completion(
                instructions="system",
                tools=ModelTools.none,
                context=(),
                output="text",
                config=OllamaChatConfig(model="ollama-test"),
                prefill=None,
            )
