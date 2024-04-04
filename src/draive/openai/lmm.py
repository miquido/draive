from typing import Literal, overload

from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam

from draive.lmm import LMMCompletionMessage, LMMCompletionStream, LMMCompletionStreamingUpdate
from draive.openai.chat_response import _chat_response  # pyright: ignore[reportPrivateUsage]
from draive.openai.chat_stream import _chat_stream  # pyright: ignore[reportPrivateUsage]
from draive.openai.client import OpenAIClient
from draive.openai.config import OpenAIChatConfig
from draive.scope import ctx
from draive.tools import Toolbox, ToolCallUpdate, ToolsUpdatesContext
from draive.types import ImageBase64Content, ImageURLContent, Model, UpdateSend
from draive.utils import AsyncStreamTask

__all__ = [
    "openai_lmm_completion",
]


@overload
async def openai_lmm_completion(
    *,
    context: list[LMMCompletionMessage],
    tools: Toolbox | None = None,
    stream: Literal[True],
) -> LMMCompletionStream:
    ...


@overload
async def openai_lmm_completion(
    *,
    context: list[LMMCompletionMessage],
    tools: Toolbox | None = None,
    stream: UpdateSend[LMMCompletionStreamingUpdate],
) -> LMMCompletionMessage:
    ...


@overload
async def openai_lmm_completion(
    *,
    context: list[LMMCompletionMessage],
    tools: Toolbox | None = None,
    output: Literal["text", "json"] = "text",
) -> LMMCompletionMessage:
    ...


async def openai_lmm_completion(
    *,
    context: list[LMMCompletionMessage],
    tools: Toolbox | None = None,
    output: Literal["text", "json"] = "text",
    stream: UpdateSend[LMMCompletionStreamingUpdate] | bool = False,
) -> LMMCompletionStream | LMMCompletionMessage:
    client: OpenAIClient = ctx.dependency(OpenAIClient)
    config: OpenAIChatConfig
    match output:
        case "text":
            config = ctx.state(OpenAIChatConfig).updated(response_format={"type": "text"})
        case "json":
            config = ctx.state(OpenAIChatConfig).updated(response_format={"type": "json_object"})
    messages: list[ChatCompletionMessageParam] = [
        _convert_message(config=config, message=message) for message in context
    ]

    match stream:
        case False:
            with ctx.nested("openai_lmm_completion", metrics=[config]):
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
                    "openai_lmm_completion",
                    state=[ToolsUpdatesContext(send_update=streaming_update)],
                    metrics=[config],
                ):

                    def send_update(update: ToolCallUpdate | str) -> None:
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

            def send_update(update: ToolCallUpdate | str) -> None:
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
                "openai_lmm_completion",
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
    config: OpenAIChatConfig,
    message: LMMCompletionMessage,
) -> ChatCompletionMessageParam:
    match message.role:
        case "user":
            if isinstance(message.content, str):
                return {
                    "role": "user",
                    "content": message.content,
                }
            elif isinstance(message.content, ImageURLContent):
                return {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": message.content.url,
                                "detail": config.vision_details,
                            },
                        }
                    ],
                }
            elif isinstance(message.content, ImageBase64Content):
                raise ValueError("Unsupported message content", message)
            elif isinstance(message.content, Model):
                return {
                    "role": "user",
                    "content": str(message.content),
                }
            else:
                content_parts: list[ChatCompletionContentPartParam] = []
                for part in message.content:
                    if isinstance(part, str):
                        content_parts.append(
                            {
                                "type": "text",
                                "text": part,
                            }
                        )
                    elif isinstance(part, ImageURLContent):
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": part.url,
                                    "detail": config.vision_details,
                                },
                            }
                        )
                    elif isinstance(message.content, ImageBase64Content):
                        raise ValueError("Unsupported message content", message)
                    elif isinstance(message.content, Model):
                        content_parts.append(
                            {
                                "type": "text",
                                "text": str(message.content),
                            }
                        )
                    else:
                        raise ValueError("Unsupported message content", message)
                return {
                    "role": "user",
                    "content": content_parts,
                }

        case "assistant":
            if isinstance(message.content, str):
                return {
                    "role": "assistant",
                    "content": message.content,
                }
            elif isinstance(message.content, Model):
                return {
                    "role": "assistant",
                    "content": str(message.content),
                }
            else:
                raise ValueError("Invalid assistant message", message)

        case "system":
            if isinstance(message.content, str):
                return {
                    "role": "system",
                    "content": message.content,
                }
            elif isinstance(message.content, Model):
                return {
                    "role": "system",
                    "content": str(message.content),
                }
            else:
                raise ValueError("Invalid system message", message)
