#! uv -n run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "draive[openai]~=0.70",
#   "pyaudio~=0.2",
#   "beautifulsoup4~=4.13",
#   "httpx~=0.28",
# ]
# ///
from asyncio import run, to_thread
from collections.abc import AsyncGenerator

import pyaudio
from bs4 import BeautifulSoup
from haiway import LoggerObservability
from haiway.utils.logs import setup_logging
from haiway.utils.queue import AsyncQueue
from httpx import AsyncClient, Response

from draive import (
    ConversationEvent,
    ConversationMessageChunk,
    Instruction,
    MediaData,
    RealtimeConversation,
    RealtimeConversationSession,
    ctx,
    load_env,
    tool,
)
from draive.openai import OpenAI, OpenAIRealtimeConfig

load_env()
setup_logging("realtime_chat")

pya = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024


@tool(description="Access content of a website")
async def website_content(url: str) -> str:
    # simplified implementation of web access
    async with AsyncClient(follow_redirects=True) as client:
        response: Response = await client.get(url)
        page = BeautifulSoup(
            markup=await response.aread(),
            features="html.parser",
        )
        return (page.find(name="main") or page.find(name="body") or page).text


async def main():
    async with ctx.scope(
        "realtime_chat",
        OpenAIRealtimeConfig(model="gpt-4o-realtime-preview"),
        disposables=(OpenAI(),),
        observability=LoggerObservability(),
    ):
        audio_replay_queue = AsyncQueue()

        async def replay_audio() -> None:
            out_audio_stream = await to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
            )
            async for audio in audio_replay_queue:
                await to_thread(
                    out_audio_stream.write,
                    audio,
                )

        ctx.spawn(replay_audio)

        async def record_audio() -> AsyncGenerator[ConversationMessageChunk]:
            mic_info = pya.get_default_input_device_info()
            audio_stream = await to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE,
            )

            from uuid import uuid4

            message_id = uuid4()

            while True:
                audio_data = MediaData.of(
                    await to_thread(audio_stream.read, CHUNK_SIZE),
                    media="audio/pcm",
                )
                yield ConversationMessageChunk.user(
                    content=audio_data,
                    message_identifier=message_id,
                )

        session_scope = await RealtimeConversation.prepare(
            instruction=Instruction.of(
                "You are helpful assistant."
                " Try to autonomously complete user requests using available tools when needed."
                " Request user help or clarification only on with ambiguity or issues."
            ),
            tools=[website_content],
        )

        async with ctx.scope("realtime", disposables=(session_scope,)):
            # Start the writer task to send audio input
            ctx.spawn(RealtimeConversationSession.writer, record_audio())

            # Read output chunks from the session
            async for chunk in RealtimeConversationSession.reader():
                if isinstance(chunk, ConversationEvent):
                    # on interrupt event clear output queue to prevent playing unnecessary content
                    if chunk.category == "interrupted":
                        audio_replay_queue.clear()

                for part in chunk.content.parts:
                    match part:
                        case MediaData() as media:
                            audio_replay_queue.enqueue(media.data)

                        case _:
                            pass  # we are expecting only audio outputs


run(main())
