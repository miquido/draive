from collections.abc import Mapping, Sequence
from typing import Any, Literal, overload

from haiway import State, ctx

from draive.commons import META_EMPTY, Meta
from draive.lmm import LMMInput
from draive.multimodal import MultimodalContent
from draive.prompts.types import (
    Prompt,
    PromptDeclaration,
    PromptFetching,
    PromptListFetching,
    PromptMissing,
)

__all__ = ("Prompts",)


class Prompts(State):
    @classmethod
    async def fetch_list(
        cls,
        **extra: Any,
    ) -> Sequence[PromptDeclaration]:
        return await ctx.state(cls).list_fetching(**extra)

    @overload
    @classmethod
    async def fetch(
        cls,
        reference: PromptDeclaration | str,
        /,
        *,
        default: Prompt | str | None = None,
        arguments: Mapping[str, str] | None = None,
        **extra: Any,
    ) -> Prompt | None: ...

    @overload
    @classmethod
    async def fetch(
        cls,
        reference: PromptDeclaration | str,
        /,
        *,
        default: Prompt | str,
        arguments: Mapping[str, str] | None = None,
        **extra: Any,
    ) -> Prompt: ...

    @overload
    @classmethod
    async def fetch(
        cls,
        reference: PromptDeclaration | str,
        /,
        *,
        default: Prompt | str | None = None,
        arguments: Mapping[str, str] | None = None,
        required: Literal[True],
        **extra: Any,
    ) -> Prompt: ...

    @classmethod
    async def fetch(
        cls,
        reference: PromptDeclaration | str,
        /,
        *,
        default: Prompt | str | None = None,
        arguments: Mapping[str, str] | None = None,
        required: bool = True,
        **extra: Any,
    ) -> Prompt | None:
        name: str = reference if isinstance(reference, str) else reference.name

        match await ctx.state(cls).fetch(
            name,
            arguments=arguments,
            **extra,
        ):
            case None:
                match default:
                    case None:
                        if required:
                            raise PromptMissing(f"Missing prompt: '{name}'")

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

    list_fetching: PromptListFetching
    fetching: PromptFetching
    meta: Meta = META_EMPTY
