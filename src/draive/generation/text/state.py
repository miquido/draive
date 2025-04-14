from collections.abc import AsyncIterable, Iterable
from typing import Any, Literal, overload

from haiway import State, ctx

from draive.generation.text.default import generate_text
from draive.generation.text.types import TextGenerating
from draive.instructions import Instruction
from draive.multimodal import Multimodal
from draive.prompts import Prompt
from draive.tools import Tool, Toolbox

__all__ = ("TextGeneration",)


class TextGeneration(State):
    @overload
    @classmethod
    async def generate(
        cls,
        *,
        instruction: Instruction | str | None = None,
        input: Prompt | Multimodal,
        tools: Toolbox | Iterable[Tool] | None = None,
        examples: Iterable[tuple[Multimodal, str]] | None = None,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> str: ...

    @overload
    @classmethod
    async def generate(
        cls,
        *,
        instruction: Instruction | str | None = None,
        input: Prompt | Multimodal,
        tools: Toolbox | Iterable[Tool] | None = None,
        examples: Iterable[tuple[Multimodal, str]] | None = None,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterable[str]: ...

    @classmethod
    async def generate(
        cls,
        *,
        instruction: Instruction | str | None = None,
        input: Prompt | Multimodal,  # noqa: A002
        tools: Toolbox | Iterable[Tool] | None = None,
        examples: Iterable[tuple[Multimodal, str]] | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterable[str] | str:
        return await ctx.state(cls).generation(
            instruction=instruction,
            input=input,
            tools=tools,
            examples=examples,
            stream=stream,
            **extra,
        )

    generation: TextGenerating = generate_text
