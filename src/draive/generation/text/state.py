from collections.abc import Iterable
from typing import Any, overload

from haiway import State, ctx, statemethod

from draive.generation.text.default import generate_text
from draive.generation.text.types import TextGenerating
from draive.models import ModelInstructions
from draive.multimodal import Multimodal, Template, TemplatesRepository
from draive.tools import Tool, Toolbox

__all__ = ("TextGeneration",)


class TextGeneration(State):
    @overload
    @classmethod
    async def generate(
        cls,
        *,
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        examples: Iterable[tuple[Multimodal, str]] = (),
        **extra: Any,
    ) -> str: ...

    @overload
    async def generate(
        self,
        *,
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        examples: Iterable[tuple[Multimodal, str]] = (),
        **extra: Any,
    ) -> str: ...

    @statemethod
    async def generate(
        self,
        *,
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,  # noqa: A002
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        examples: Iterable[tuple[Multimodal, str]] = (),
        **extra: Any,
    ) -> str:
        """Generate text using the configured generative model.

        Parameters
        ----------
        instructions : Template | ModelInstructions, default=""
            Instructions or a template resolved before generating.
        input : Template | Multimodal
            Input content or a template resolved before generating.
        tools : Toolbox | Iterable[Tool], default=Toolbox.empty
            Tools available to the model while generating.
        examples : Iterable[tuple[Multimodal, str]], default=()
            Example input and output pairs provided before the input.
        **extra : Any
            Additional provider options forwarded to completion.

        Returns
        -------
        str
            Generated text.
        """
        async with ctx.scope("text_generation"):
            if isinstance(instructions, Template):
                ctx.record_info(
                    attributes={"instructions.template": instructions.identifier},
                )
                instructions = await TemplatesRepository.resolve_str(instructions)

            if isinstance(input, Template):
                ctx.record_info(
                    attributes={"input.template": input.identifier},
                )
                input = await TemplatesRepository.resolve(input)  # noqa: A001

            return await self._generating(
                instructions=instructions,
                input=input,
                toolbox=Toolbox.of(tools),
                examples=examples,
                **extra,
            )

    _generating: TextGenerating = generate_text

    def __init__(
        self,
        generating: TextGenerating = generate_text,
    ) -> None:
        super().__init__(_generating=generating)
