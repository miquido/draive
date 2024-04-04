from collections.abc import Iterable
from typing import TypeVar

from draive.lmm import LMMCompletionMessage, lmm_completion
from draive.tools import Toolbox
from draive.types import Model, MultimodalContent

__all__: list[str] = [
    "lmm_generate_model",
]

_Generated = TypeVar(
    "_Generated",
    bound=Model,
)


async def lmm_generate_model(
    model: type[_Generated],
    *,
    instruction: str,
    input: MultimodalContent,  # noqa: A002
    tools: Toolbox | None = None,
    examples: Iterable[tuple[MultimodalContent, _Generated]] | None = None,
) -> _Generated:
    system_message: LMMCompletionMessage = LMMCompletionMessage(
        role="system",
        content=INSTRUCTION.format(
            instruction=instruction,
            format=model.specification_json(),
        ),
    )
    input_message: LMMCompletionMessage = LMMCompletionMessage(
        role="user",
        content=input,
    )

    context: list[LMMCompletionMessage]

    if examples:
        context = [
            system_message,
            *[
                message
                for example in examples
                for message in [
                    LMMCompletionMessage(
                        role="user",
                        content=example[0],
                    ),
                    LMMCompletionMessage(
                        role="assistant",
                        content=example[1].as_json(indent=2),
                    ),
                ]
            ],
            input_message,
        ]

    else:
        context = [
            system_message,
            input_message,
        ]

    completion: LMMCompletionMessage = await lmm_completion(
        context=context,
        tools=tools,
        output="json",
    )
    generated: _Generated = model.from_json(completion.content_string)

    return generated


INSTRUCTION: str = """\
{instruction}

IMPORTANT!
The result have to conform to the following JSON Schema:
```
{format}
```
Provide ONLY a single, raw, valid JSON without any comments or formatting.
"""
