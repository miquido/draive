from typing import Literal, overload

from mistralai.models.chat_completion import ChatMessage

from draive.lmm import LMMCompletionMessage, LMMCompletionStream, LMMCompletionStreamingUpdate
from draive.mistral.chat_response import _chat_response  # pyright: ignore[reportPrivateUsage]
from draive.mistral.chat_stream import _chat_stream  # pyright: ignore[reportPrivateUsage]
from draive.mistral.client import MistralClient
from draive.mistral.config import MistralChatConfig
from draive.scope import ctx
from draive.tools import Toolbox, ToolCallUpdate, ToolsUpdatesContext
from draive.types import ImageBase64Content, ImageURLContent, Model, UpdateSend
from draive.utils import AsyncStreamTask

__all__ = [
    "mistral_lmm_completion",
]


@overload
async def mistral_lmm_completion(
    *,
    context: list[LMMCompletionMessage],
    tools: Toolbox | None = None,
    stream: Literal[True],
) -> LMMCompletionStream:
    ...


@overload
async def mistral_lmm_completion(
    *,
    context: list[LMMCompletionMessage],
    tools: Toolbox | None = None,
    stream: UpdateSend[LMMCompletionStreamingUpdate],
) -> LMMCompletionMessage:
    ...


@overload
async def mistral_lmm_completion(
    *,
    context: list[LMMCompletionMessage],
    tools: Toolbox | None = None,
    output: Literal["text", "json"] = "text",
) -> LMMCompletionMessage:
    ...


async def mistral_lmm_completion(
    *,
    context: list[LMMCompletionMessage],
    tools: Toolbox | None = None,
    output: Literal["text", "json"] = "text",
    stream: UpdateSend[LMMCompletionStreamingUpdate] | bool = False,
) -> LMMCompletionStream | LMMCompletionMessage:
    client: MistralClient = ctx.dependency(MistralClient)
    config: MistralChatConfig
    match output:
        case "text":
            config = ctx.state(MistralChatConfig).updated(response_format={"type": "text"})
        case "json":
            if tools is None:
                config = ctx.state(MistralChatConfig).updated(
                    response_format={"type": "json_object"}
                )
            else:
                ctx.log_warning(
                    "Attempting to use Mistral in JSON mode with tools which is not supported."
                    " Using text mode instead..."
                )
                config = ctx.state(MistralChatConfig).updated(response_format={"type": "text"})
    messages: list[ChatMessage] = [_convert_message(message=message) for message in context]

    match stream:
        case False:
            with ctx.nested("mistral_lmm_completion", metrics=[config]):
                return LMMCompletionMessage(
                    role="assistant",
                    content=await _chat_response(
                        client=client,
                        config=config,
                        messages=messages,
                        tools=tools,
                    ),
                )

        case True:

            async def stream_task(
                streaming_update: UpdateSend[LMMCompletionStreamingUpdate],
            ) -> None:
                with ctx.nested(
                    "mistral_lmm_completion",
                    state=[ToolsUpdatesContext(send_update=streaming_update)],
                    metrics=[config],
                ):

                    def send_update(update: ToolCallUpdate | str):
                        if isinstance(update, str):
                            streaming_update(
                                LMMCompletionMessage(
                                    role="assistant",
                                    content=update,
                                )
                            )
                        else:
                            streaming_update(update)

                    await _chat_stream(
                        client=client,
                        config=config,
                        messages=messages,
                        tools=tools,
                        send_update=send_update,
                    )

            return AsyncStreamTask(job=stream_task)

        case streaming_update:

            def send_update(update: ToolCallUpdate | str):
                if isinstance(update, str):
                    streaming_update(
                        LMMCompletionMessage(
                            role="assistant",
                            content=update,
                        )
                    )
                else:
                    streaming_update(update)

            with ctx.nested(
                "mistral_lmm_completion",
                state=[ToolsUpdatesContext(send_update=streaming_update)],
                metrics=[config],
            ):
                return LMMCompletionMessage(
                    role="assistant",
                    content=await _chat_stream(
                        client=client,
                        config=config,
                        messages=messages,
                        tools=tools,
                        send_update=send_update,
                    ),
                )


def _convert_message(  # noqa: PLR0912, C901, PLR0911
    message: LMMCompletionMessage,
) -> ChatMessage:
    match message.role:
        case "user":
            if isinstance(message.content, str):
                return ChatMessage(
                    role="user",
                    content=message.content,
                )
            elif isinstance(message.content, ImageURLContent):
                raise ValueError("Unsupported message content", message)
            elif isinstance(message.content, ImageBase64Content):
                raise ValueError("Unsupported message content", message)
            elif isinstance(message.content, Model):
                return ChatMessage(
                    role="user",
                    content=str(message.content),
                )
            else:
                content_parts: list[str] = []
                for part in message.content:
                    if isinstance(part, str):
                        content_parts.append(part)
                    elif isinstance(part, ImageURLContent):
                        raise ValueError("Unsupported message content", message)
                    elif isinstance(message.content, ImageBase64Content):
                        raise ValueError("Unsupported message content", message)
                    elif isinstance(message.content, Model):
                        content_parts.append(str(message.content))
                    else:
                        raise ValueError("Unsupported message content", message)
                return ChatMessage(
                    role="user",
                    content=content_parts,
                )

        case "assistant":
            if isinstance(message.content, str):
                return ChatMessage(
                    role="assistant",
                    content=message.content,
                )
            elif isinstance(message.content, Model):
                return ChatMessage(
                    role="assistant",
                    content=str(message.content),
                )
            else:
                raise ValueError("Invalid assistant message", message)

        case "system":
            if isinstance(message.content, str):
                return ChatMessage(
                    role="system",
                    content=message.content,
                )
            elif isinstance(message.content, Model):
                return ChatMessage(
                    role="system",
                    content=str(message.content),
                )
            else:
                raise ValueError("Invalid system message", message)
