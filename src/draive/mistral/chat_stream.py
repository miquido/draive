from collections.abc import Callable

from draive.mistral.chat_response import _chat_response  # pyright: ignore[reportPrivateUsage]
from draive.mistral.client import MistralClient
from draive.mistral.config import MistralChatConfig
from draive.mistral.models import ChatMessage
from draive.scope import ctx
from draive.tools import Toolbox, ToolCallUpdate

__all__ = [
    "_chat_stream",
]


async def _chat_stream(  # noqa: PLR0913
    *,
    client: MistralClient,
    config: MistralChatConfig,
    messages: list[ChatMessage],
    tools: Toolbox,
    send_update: Callable[[ToolCallUpdate | str], None],
    recursion_level: int = 0,
) -> str:
    ctx.log_warning("Mistral streaming api is not supported yet, using regular response...")
    message: str = await _chat_response(
        client=client,
        config=config,
        messages=messages,
        tools=tools,
        recursion_level=recursion_level,
    )
    send_update(message)
    return message
