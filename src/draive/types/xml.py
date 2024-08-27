from collections.abc import Callable, Generator
from typing import cast, overload

from draive.types.multimodal import Multimodal, MultimodalContent, MultimodalContentElement
from draive.types.text import TextContent

__all__ = [
    "xml_tag",
    "xml_tags",
]


@overload
def xml_tags(
    tag: str,
    /,
    source: Multimodal,
) -> Generator[MultimodalContent, None]: ...


@overload
def xml_tags[Result](
    tag: str,
    /,
    source: Multimodal,
    conversion: Callable[[MultimodalContent], Result],
) -> Generator[Result, None]: ...


def xml_tags[Result](  # noqa: C901, PLR0912, PLR0915
    tag: str,
    /,
    source: Multimodal,
    conversion: Callable[[MultimodalContent], Result] | None = None,
) -> Generator[MultimodalContent, None] | Generator[Result, None]:
    content_parts: tuple[MultimodalContentElement, ...]
    match source:
        case str() as string:
            content_parts = (TextContent(text=string),)

        case MultimodalContent() as multimodal:
            content_parts = multimodal.parts

        case TextContent() as text:
            content_parts = (text,)

        case _:
            return  # can't process other types

    opening_tag_prefix: str = f"<{tag} "
    opening_tag: str = f"<{tag}>"
    closing_tag: str = f"</{tag}>"

    def check_opening(
        accumulator: str,
        /,
    ) -> bool:
        return accumulator == opening_tag or (
            accumulator.endswith(">") and accumulator.startswith(opening_tag_prefix)
        )

    accumulator: str = ""
    content_accumulator: str = ""
    content_accumulator_meta: dict[str, str | float | int | bool | None] | None = None
    content: list[MultimodalContentElement] = []
    in_tag: bool = False
    in_content: bool = False
    for part in content_parts:
        match part:
            case TextContent() as text:
                # preserve parts as in input
                if in_content and content_accumulator:
                    content.append(
                        TextContent(
                            text=content_accumulator,
                            meta=content_accumulator_meta,
                        )
                    )
                    content_accumulator = ""

                content_accumulator_meta = text.meta  # use current part meta

                for char in text.text:
                    if in_tag:
                        if char == ">":
                            accumulator += char
                            if in_content and accumulator == closing_tag:
                                in_content = False
                                if conversion := conversion:
                                    if content_accumulator:
                                        yield conversion(
                                            MultimodalContent.of(
                                                *content,
                                                TextContent(
                                                    text=content_accumulator,
                                                    meta=content_accumulator_meta,
                                                ),
                                            )
                                        )

                                    else:
                                        yield conversion(MultimodalContent.of(*content))

                                elif content_accumulator:
                                    yield MultimodalContent.of(
                                        *content,
                                        TextContent(
                                            text=content_accumulator,
                                            meta=content_accumulator_meta,
                                        ),
                                    )

                                else:
                                    yield MultimodalContent.of(*content)

                                content = []  # clear to be ready for next tag
                                content_accumulator = ""

                            elif check_opening(accumulator):
                                in_content = True
                                content = []  # clear current content in case of nested tags
                                content_accumulator = ""

                            elif in_content:
                                content_accumulator += accumulator

                            in_tag = False
                            accumulator = ""

                        elif char == "<":
                            if in_content:
                                content_accumulator += accumulator

                            accumulator = char

                        else:
                            accumulator += char

                    elif char == "<":
                        in_tag = True
                        accumulator = char

                    elif in_content:
                        content_accumulator += char

                    # else skip character

            case other:
                if in_content:
                    if accumulator:
                        content_accumulator += accumulator

                    if content_accumulator:
                        content.append(
                            TextContent(
                                text=content_accumulator,
                                meta=content_accumulator_meta,
                            )
                        )
                        content_accumulator = ""

                    content.append(other)

                if in_tag:
                    in_tag = False
                    accumulator = ""


@overload
def xml_tag(
    tag: str,
    /,
    source: Multimodal,
) -> MultimodalContent | None: ...


@overload
def xml_tag[Result](
    tag: str,
    /,
    source: Multimodal,
    conversion: Callable[[MultimodalContent], Result],
) -> Result | None: ...


def xml_tag[Result](
    tag: str,
    /,
    source: Multimodal,
    conversion: Callable[[MultimodalContent], Result] | None = None,
) -> MultimodalContent | Result | None:
    try:
        return cast(
            MultimodalContent | Result,
            next(
                xml_tags(  # pyright: ignore[reportUnknownArgumentType]
                    tag,
                    source=source,
                    conversion=conversion,  # pyright: ignore[reportArgumentType]
                ),
                None,
            ),
        )

    except StopIteration:
        return None
