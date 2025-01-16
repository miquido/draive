from collections.abc import Mapping, Sequence
from typing import Any, Literal, overload

from haiway import ctx

from draive.lmm import LMMInput
from draive.multimodal import MultimodalContent
from draive.prompts.state import PromptRepository
from draive.prompts.types import MissingPrompt, Prompt, PromptDeclaration

__all__ = [
    "fetch_prompt",
    "fetch_prompt_list",
]


async def fetch_prompt_list(
    **extra: Any,
) -> Sequence[PromptDeclaration]:
    return await ctx.state(PromptRepository).list(**extra)


@overload
async def fetch_prompt(
    reference: PromptDeclaration | str,
    /,
    *,
    default: Prompt | str | None = None,
    arguments: Mapping[str, str] | None = None,
    **extra: Any,
) -> Prompt | None: ...


@overload
async def fetch_prompt(
    reference: PromptDeclaration | str,
    /,
    *,
    default: Prompt | str,
    arguments: Mapping[str, str] | None = None,
    **extra: Any,
) -> Prompt: ...


@overload
async def fetch_prompt(
    reference: PromptDeclaration | str,
    /,
    *,
    default: Prompt | str | None = None,
    arguments: Mapping[str, str] | None = None,
    required: Literal[True],
    **extra: Any,
) -> Prompt: ...


async def fetch_prompt(
    reference: PromptDeclaration | str,
    /,
    *,
    default: Prompt | str | None = None,
    arguments: Mapping[str, str] | None = None,
    required: bool = True,
    **extra: Any,
) -> Prompt | None:
    name: str = reference if isinstance(reference, str) else reference.name

    match await ctx.state(PromptRepository).fetch(
        name,
        arguments=arguments,
        **extra,
    ):
        case None:
            match default:
                case None:
                    if required:
                        raise MissingPrompt(f"Missing prompt: '{name}'")

                    else:
                        return None

                case Prompt() as prompt:
                    return prompt

                case str() as text:
                    return Prompt.of(
                        LMMInput(
                            content=MultimodalContent.of(
                                text.format_map(arguments) if arguments else text
                            )
                        ),
                        name=name,
                        description=reference.description
                        if isinstance(reference, Prompt)
                        else None,
                    )

        case prompt:
            return prompt
