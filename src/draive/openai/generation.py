from collections.abc import Iterable
from typing import TypeVar

from draive.openai.chat import openai_chat_completion
from draive.openai.config import OpenAIChatConfig
from draive.scope import ctx
from draive.types import ConversationMessage, Model, MultimodalContent, Toolset

__all__ = [
    "openai_generate_model",
    "openai_generate_text",
]


async def openai_generate_text(
    *,
    instruction: str,
    input: MultimodalContent,  # noqa: A002
    toolset: Toolset | None = None,
    examples: Iterable[tuple[MultimodalContent, str]] | None = None,
) -> str:
    return await openai_chat_completion(
        config=ctx.state(OpenAIChatConfig),
        instruction=instruction,
        input=input,
        history=(
            [
                message
                for example in examples
                for message in [
                    ConversationMessage(
                        role="user",
                        content=example[0],
                    ),
                    ConversationMessage(
                        role="assistant",
                        content=example[1],
                    ),
                ]
            ]
            if examples
            else None
        ),
        toolset=toolset,
    )


INSTRUCTION: str = """\
{instruction}


IMPORTANT!
The result have to be a single, valid JSON without any comments or additions. \
The result have to conform to the following JSON Schema:
```
{format}
```
"""


_Generated = TypeVar(
    "_Generated",
    bound=Model,
)


async def openai_generate_model(
    model: type[_Generated],
    *,
    instruction: str,
    input: MultimodalContent,  # noqa: A002
    toolset: Toolset | None = None,
    examples: Iterable[tuple[MultimodalContent, _Generated]] | None = None,
) -> _Generated:
    return model.from_json(
        value=await openai_chat_completion(
            config=ctx.state(OpenAIChatConfig).updated(
                response_format={"type": "json_object"},
            ),
            instruction=INSTRUCTION.format(
                instruction=instruction,
                format=model.specification_json(),
            ),
            input=input,
            history=(
                [
                    message
                    for example in examples
                    for message in [
                        ConversationMessage(
                            role="user",
                            content=example[0],
                        ),
                        ConversationMessage(
                            role="assistant",
                            content=example[1].as_json(),
                        ),
                    ]
                ]
                if examples
                else None
            ),
            toolset=toolset,
        )
    )
