from collections.abc import Iterable
from typing import TypeVar

from draive.mistral.chat import mistral_chat_completion
from draive.mistral.config import MistralChatConfig
from draive.scope import ctx
from draive.types import ConversationMessage, Model, StringConvertible, Toolset

__all__ = [
    "mistral_generate",
    "mistral_generate_text",
]

_Generated = TypeVar(
    "_Generated",
    bound=Model,
)

INSTRUCTION: str = """\
{instruction}

The output have to be formatted as a JSON object that conforms to the following JSON Schema:
```
{format}
```
"""


async def mistral_generate_text(
    *,
    instruction: str,
    input: StringConvertible,  # noqa: A002
    toolset: Toolset | None = None,
    examples: Iterable[tuple[str, str]] | None = None,
) -> str:
    return await mistral_chat_completion(
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


async def mistral_generate(
    model: type[_Generated],
    *,
    instruction: str,
    input: StringConvertible,  # noqa: A002
    toolset: Toolset | None = None,
    examples: Iterable[tuple[str, _Generated]] | None = None,
) -> _Generated:
    with ctx.updated(
        ctx.state(MistralChatConfig).updated(
            response_format={"type": "json_object"},
        )
    ):
        return model.from_json(
            value=await mistral_chat_completion(
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
                                content=example[1].__str__(),
                            ),
                        ]
                    ]
                    if examples
                    else None
                ),
                toolset=toolset,
            )
        )
