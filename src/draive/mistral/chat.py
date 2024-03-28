# from typing import Literal, overload

# from mistralai.models.chat_completion import ChatMessage

# from draive.mistral.chat_response import _chat_response  # pyright: ignore[reportPrivateUsage]
# from draive.mistral.chat_stream import _chat_stream  # pyright: ignore[reportPrivateUsage]
# from draive.mistral.client import MistralClient
# from draive.mistral.config import MistralChatConfig
# from draive.scope import ctx
# from draive.tools import ToolsUpdatesContext
# from draive.types import (
#     ConversationMessage,
#     ConversationMessageContent,
#     ConversationResponseStream,
#     ConversationStreamingUpdate,
#     Toolbox,
#     UpdateSend,
# )
# from draive.utils import AsyncStreamTask

# __all__ = [
#     "mistral_chat_completion",
# ]


# @overload
# async def mistral_chat_completion(
#     *,
#     config: MistralChatConfig,
#     instruction: str,
#     input: ConversationMessage | ConversationMessageContent,
#     history: list[ConversationMessage] | None = None,
#     tools: Toolbox | None = None,
#     stream: Literal[True],
# ) -> ConversationResponseStream:
#     ...


# @overload
# async def mistral_chat_completion(
#     *,
#     config: MistralChatConfig,
#     instruction: str,
#     input: ConversationMessage | ConversationMessageContent,
#     history: list[ConversationMessage] | None = None,
#     tools: Toolbox | None = None,
#     stream: UpdateSend[ConversationStreamingUpdate],
# ) -> str:
#     ...


# @overload
# async def mistral_chat_completion(
#     *,
#     config: MistralChatConfig,
#     instruction: str,
#     input: ConversationMessage | ConversationMessageContent,
#     history: list[ConversationMessage] | None = None,
#     tools: Toolbox | None = None,
# ) -> str:
#     ...


# async def mistral_chat_completion(
#     *,
#     config: MistralChatConfig,
#     instruction: str,
#     input: ConversationMessage | ConversationMessageContent,
#     history: list[ConversationMessage] | None = None,
#     tools: Toolbox | None = None,
#     stream: UpdateSend[ConversationStreamingUpdate] | bool = False,
# ) -> ConversationResponseStream | str:
#     with ctx.nested("mistral_chat_completion", config):
#         client: MistralClient = ctx.dependency(MistralClient)
#         messages: list[ChatMessage] = _prepare_messages(
#             instruction=instruction,
#             history=history or [],
#             input=input,
#             limit=config.context_messages_limit,
#         )
#         match stream:
#             case False:
#                 return await _chat_response(
#                     client=client,
#                     config=config,
#                     messages=messages,
#                     tools=toolset,
#                 )

#             case True:

#                 async def stream_task(
#                     progress: UpdateSend[ConversationStreamingUpdate],
#                 ) -> None:
#                     with ctx.updated(
#                         ToolsUpdatesContext(
#                             send_update=progress or (lambda update: None),
#                         ),
#                     ):
#                         await _chat_stream(
#                             client=client,
#                             config=config,
#                             messages=messages,
#                             tools=toolset,
#                             updates=progress,
#                         )

#                 return AsyncStreamTask(job=stream_task)

#             case progress:
#                 with ctx.updated(
#                     ToolsUpdatesContext(
#                         send_update=progress or (lambda update: None),
#                     ),
#                 ):
#                     return await _chat_stream(
#                         client=client,
#                         config=config,
#                         messages=messages,
#                         tools=toolset,
#                         updates=progress,
#                     )


# def _prepare_messages(
#     instruction: str,
#     history: list[ConversationMessage],
#     input: ConversationMessage | ConversationMessageContent,
#     limit: int,
# ) -> list[ChatMessage]:
#     assert limit > 0, "Messages limit has to be greater than zero"  # nosec: B101
#     input_message: ChatMessage
#     if isinstance(input, ConversationMessage):
#         input_message = _convert_message(message=input)
#     else:
#         input_message = _convert_message(
#             message=ConversationMessage(
#                 role="user",
#                 content=input,
#             )
#         )

#     messages: list[ChatMessage] = []
#     for message in history:
#         try:
#             messages.append(_convert_message(message=message))
#         except ValueError:
#             ctx.log_error(
#                 "Invalid message: %s Ignoring memory.",
#                 message,
#             )
#             return [
#                 ChatMessage(
#                     role="system",
#                     content=instruction,
#                 ),
#                 input_message,
#             ]
#     return [
#         ChatMessage(
#             role="system",
#             content=instruction,
#         ),
#         *messages[-limit:],
#         input_message,
#     ]


# def _convert_message(
#     message: ConversationMessage,
# ) -> ChatMessage:
#     match message.role:
#         case "user":
#             if isinstance(message.content, str):
#                 return ChatMessage(
#                     role="user",
#                     content=message.content,
#                 )
#             elif isinstance(message.content, list):
#                 content_parts: list[str] = []
#                 for part in message.content:
#                     if isinstance(part, str):
#                         content_parts.append(part)
#                     else:
#                         raise ValueError("Unsupported message content", message)
#                 return ChatMessage(
#                     role="user",
#                     content="\n".join(content_parts),
#                 )
#             else:
#                 raise ValueError("Unsupported message content", message)

#         case "assistant":
#             if isinstance(message.content, str):
#                 return ChatMessage(
#                     role="assistant",
#                     content=message.content,
#                 )
#             else:
#                 raise ValueError("Invalid assistant message", message)

#         case other:
#             raise ValueError("Invalid message role", other)
