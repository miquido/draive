from collections.abc import Generator, Iterator, Mapping
from typing import Self, overload

from haiway import State

from draive.commons import META_EMPTY, Meta
from draive.multimodal.content import Multimodal, MultimodalContent, MultimodalContentElement
from draive.multimodal.text import TextContent
from draive.parameters import DataModel

__all__ = ("MultimodalTagElement",)


# TODO: add streaming support with content input as async iterator
class MultimodalTagElement(DataModel):
    name: str
    content: MultimodalContent
    attributes: Mapping[str, str] | None = None

    @overload
    @classmethod
    def parse_first(
        cls,
        tag: str | None = None,
        /,
        *,
        content: Multimodal,
    ) -> Self | None: ...

    @overload
    @classmethod
    def parse_first(
        cls,
        tag: str | None = None,
        /,
        *,
        content: Multimodal,
        default: Self,
    ) -> Self: ...

    @classmethod
    def parse_first(
        cls,
        tag: str | None = None,
        /,
        *,
        content: Multimodal,
        default: Self | None = None,
    ) -> Self | None:
        return next(
            cls.parse(
                tag,
                content=content,
            ),
            default,
        )

    @classmethod
    def parse(  # noqa: C901, PLR0912
        cls,
        tag: str | None = None,
        /,
        *,
        content: Multimodal,
    ) -> Generator[Self]:
        current_tag: _TagOpening | None = None
        current_tag_content: list[MultimodalContentElement] = []
        for part in MultimodalContent.of(content).parts:
            match part:
                case TextContent() as text_content:
                    text_iterator = _CharactersIterator(text_content.text)
                    text_accumulator: str = ""  # accumulator for any parts that are not tags
                    while text_iterator:  # keep working until end of text part
                        if current_tag is None:  # find the tag opening first
                            for element in _parse_tag_opening(
                                text_iterator,
                                meta=text_content.meta,
                            ):
                                match element:
                                    case _TagOpening() as opening:
                                        # check if tag is already closed
                                        if opening.closed:
                                            # check if it is the tag we are looking for
                                            if not tag or opening.name == tag:
                                                yield cls(
                                                    name=opening.name,
                                                    attributes=opening.attributes,
                                                    content=MultimodalContent.empty,
                                                )

                                            # cleanup regardless of matching
                                            current_tag = None
                                            continue  # continue the loop

                                        else:
                                            # begin new tag
                                            current_tag = opening
                                            # cleanup content and accumulator
                                            current_tag_content = []
                                            break  # break the loop and find tag closing

                                    case _:
                                        pass  # skip other elements - we are out of any tag

                        if current_tag is not None:  # find the tag closing if needed
                            for element in _parse_tag_closing(
                                text_iterator,
                                tag=current_tag,
                                meta=text_content.meta,
                            ):
                                match element:
                                    case _TagClosing():
                                        if text_accumulator:
                                            # append accumulated content if needed
                                            current_tag_content.append(
                                                TextContent(
                                                    text=text_accumulator,
                                                    meta=text_content.meta,
                                                )
                                            )
                                            text_accumulator = ""  # cleanup

                                        # check if it is the tag we are looking for
                                        if not tag or current_tag.name == tag:
                                            yield cls(
                                                name=current_tag.name,
                                                attributes=current_tag.attributes,
                                                content=MultimodalContent.of(*current_tag_content),
                                            )
                                        # we have closed current tag - cleanup
                                        current_tag = None
                                        current_tag_content = []
                                        break  # and break the loop - we have closing

                                    case text:
                                        text_accumulator += text

                        continue  # continue until the end of text part

                    # add text reminder to tag content at the end of the element
                    if current_tag is not None and text_accumulator:
                        current_tag_content.append(
                            TextContent(
                                text=text_accumulator,
                                meta=text_content.meta,
                            )
                        )

                case other:  # media or artifact
                    if current_tag is not None:
                        # include in tag content
                        current_tag_content.append(other)

    @classmethod
    def replace(  # noqa: C901, PLR0912, PLR0915
        cls,
        tag: str,
        /,
        *,
        content: Multimodal,
        replacement: Multimodal,
        strip_tags: bool = False,
        replace_all: bool = False,
    ) -> MultimodalContent:
        # TODO: add support for tag id attribute or custom matching?
        replacement_content = MultimodalContent.of(replacement).parts
        result_content: list[MultimodalContentElement] = []
        current_tag: _TagOpening | None = None
        current_tag_content: list[MultimodalContentElement] = []
        replaced: bool = False
        for part in MultimodalContent.of(content).parts:
            # check if we need to replace anything more
            if replaced and not replace_all:
                if current_tag is not None:
                    # include leftovers
                    result_content.extend(
                        (
                            TextContent(
                                text=current_tag.raw,
                                meta=current_tag.meta,
                            ),
                            MultimodalContent.of(*current_tag_content),
                        ),
                    )
                    current_tag = None
                    current_tag_content = []

                result_content.append(part)
                continue  # skip parsing

            match part:
                case TextContent() as text_content:
                    text_iterator = _CharactersIterator(text_content.text)
                    text_accumulator: str = ""  # accumulator for any parts that are not tags
                    while text_iterator:  # keep working until end of text part
                        if current_tag is None:  # find the tag opening first
                            for element in _parse_tag_opening(
                                text_iterator,
                                meta=text_content.meta,
                            ):
                                match element:
                                    case _TagOpening() as opening:
                                        # include all content up to this point
                                        if text_accumulator:
                                            result_content.append(
                                                TextContent(
                                                    text=text_accumulator,
                                                    meta=text_content.meta,
                                                )
                                            )
                                            text_accumulator = ""  # cleanup

                                        # check if tag is already closed
                                        if opening.closed:
                                            # check if it is the tag we are looking for
                                            if (not tag or opening.name == tag) and (
                                                replace_all or not replaced
                                            ):
                                                replaced = True

                                                if strip_tags:
                                                    result_content.extend(replacement_content)

                                                else:
                                                    result_content.extend(
                                                        (  # expand tag
                                                            TextContent(
                                                                text=opening.raw.removesuffix("/>")
                                                                + ">",
                                                                meta=opening.meta,
                                                            )
                                                            if opening.closed
                                                            else TextContent(
                                                                text=opening.raw,
                                                                meta=opening.meta,
                                                            ),
                                                            *replacement_content,
                                                            TextContent(
                                                                text=f"</{opening.name}>",
                                                                meta=opening.meta,
                                                            ),
                                                        ),
                                                    )
                                            else:
                                                result_content.append(
                                                    TextContent(
                                                        text=opening.raw,
                                                        meta=opening.meta,
                                                    ),
                                                )

                                            # cleanup regardless of matching
                                            current_tag = None
                                            continue  # continue the loop

                                        else:
                                            # begin new tag
                                            current_tag = opening
                                            # cleanup content
                                            current_tag_content = []
                                            break  # break the loop and find tag closing

                                    case other:
                                        text_accumulator += other

                        if current_tag is not None:  # find the tag closing if needed
                            for element in _parse_tag_closing(
                                text_iterator,
                                tag=current_tag,
                                meta=text_content.meta,
                            ):
                                match element:
                                    case _TagClosing() as closing:
                                        # check if we should replace
                                        if (not tag or current_tag.name == tag) and (
                                            replace_all or not replaced
                                        ):
                                            replaced = True

                                            if strip_tags:
                                                result_content.extend(replacement_content)

                                            else:
                                                result_content.extend(
                                                    (  # expand tag
                                                        TextContent(
                                                            text=current_tag.raw.removesuffix("/>")
                                                            + ">",
                                                            meta=current_tag.meta,
                                                        )
                                                        if current_tag.closed
                                                        else TextContent(
                                                            text=current_tag.raw,
                                                            meta=current_tag.meta,
                                                        ),
                                                        *replacement_content,
                                                        TextContent(
                                                            text=closing.raw,
                                                            meta=closing.meta,
                                                        ),
                                                    ),
                                                )

                                        else:
                                            result_content.extend(
                                                (  # expand tag
                                                    TextContent(
                                                        text=current_tag.raw.removesuffix("/>")
                                                        + ">",
                                                        meta=current_tag.meta,
                                                    )
                                                    if current_tag.closed
                                                    else TextContent(
                                                        text=current_tag.raw,
                                                        meta=current_tag.meta,
                                                    ),
                                                    *current_tag_content,
                                                    TextContent(  # include accumulator leftover
                                                        text=text_accumulator,
                                                        meta=text_content.meta,
                                                    ),
                                                    TextContent(
                                                        text=closing.raw,
                                                        meta=closing.meta,
                                                    ),
                                                ),
                                            )

                                        # skip unwanted tags and clear out
                                        current_tag = None
                                        current_tag_content = []
                                        text_accumulator = ""
                                        break  # and break the loop - we have closing

                                    case text:
                                        text_accumulator += text

                        continue  # continue until end of part text

                    # add text reminder to tag or result content at the end of the element
                    if text_accumulator:
                        reminder_part = TextContent(
                            text=text_accumulator,
                            meta=text_content.meta,
                        )

                        if current_tag is None:
                            result_content.append(reminder_part)

                        else:
                            current_tag_content.append(reminder_part)

                case other:  # media or artifact
                    if current_tag is None:
                        # include in result
                        result_content.append(other)

                    else:
                        # include in tag content
                        current_tag_content.append(other)

        if current_tag is not None:
            # include leftovers
            result_content.extend(
                (
                    TextContent(
                        text=current_tag.raw,
                        meta=current_tag.meta,
                    ),
                    MultimodalContent.of(*current_tag_content),
                ),
            )
        return MultimodalContent.of(*result_content)


class _TagOpening(State):
    name: str
    attributes: Mapping[str, str] | None
    raw: str
    meta: Meta = META_EMPTY
    closed: bool


# look for tag opening and pass through everything else
def _parse_tag_opening(  # noqa: C901, PLR0912
    source: Iterator[str],
    /,
    meta: Meta,
) -> Generator[_TagOpening | str]:
    accumulator: str | None = None  # None is out of tag, empty is a started tag
    attributes: list[tuple[str, str]] = []
    closed: bool = False
    while character := next(source, None):
        if accumulator is None:  # check if we started a tag
            match character:
                case "<":  # potential opening
                    # prepare accumulators for new tag
                    accumulator = ""
                    attributes = []
                    closed = False

                case char:  # skip anything else
                    yield char

        else:
            match character:
                case ">" if accumulator:  # closing
                    yield _TagOpening(
                        name=accumulator,
                        attributes=dict(attributes) if attributes else None,
                        raw=(
                            "<"
                            + accumulator
                            + _escaped_attributes(attributes)
                            + ("/>" if closed else ">")
                        ),
                        meta=meta,
                        closed=closed,
                    )
                    return  # end of parsing

                case char if str.isalnum(char) or char == "_":  # part of the tag name
                    accumulator += char

                case "/" if accumulator and not closed:  # possible closed tag
                    closed = True

                case _ if (
                    str.isspace(char) and not closed and accumulator
                ):  # possible attribute start
                    for element in _parse_tag_attribute(source):
                        match element:
                            case tuple() as attribute:
                                attributes.append(attribute)

                            case ">":  # closing
                                yield _TagOpening(
                                    name=accumulator,
                                    attributes=dict(attributes) if attributes else None,
                                    raw=(
                                        "<"
                                        + accumulator
                                        + _escaped_attributes(attributes)
                                        + ("/>" if closed else ">")
                                    ),
                                    meta=meta,
                                    closed=closed,
                                )
                                return  # end of parsing

                            case _ if str.isspace(element):
                                # TODO: we are not preserving spaces correctly
                                continue  # possible next attribute

                            case "/" if not closed:  # possible closed tag
                                closed = True
                                break  # FIXME: the only valid successor is closing??

                            case other:
                                # we are no longer in tag processing, clear out
                                yield (
                                    "<"
                                    + accumulator
                                    + _escaped_attributes(attributes)
                                    + " "
                                    + other
                                )
                                accumulator = None
                                attributes = []
                                closed = False
                                break  # we are no longer looking for attributes

                case "<":  # potential new tag opening
                    # clear accumulators and start fresh
                    yield (
                        "<"
                        + accumulator
                        + _escaped_attributes(attributes)
                        + ("/" if closed else "")
                    )
                    accumulator = ""
                    attributes = []
                    closed = False

                case other:  # anything else
                    # we are no longer in tag processing, clear out
                    yield (
                        "<"
                        + accumulator
                        + _escaped_attributes(attributes)
                        + ("/" if closed else "")
                        + other
                    )
                    accumulator = None
                    attributes = []
                    closed = False

    if accumulator is not None:
        yield ("<" + accumulator + _escaped_attributes(attributes) + ("/" if closed else ""))


def _parse_tag_attribute(
    source: Iterator[str],
    /,
) -> Generator[tuple[str, str] | str]:
    while source:
        match _parse_tag_attribute_name(source):
            case (attribute,):
                match _parse_tag_attribute_value(source):
                    case (value,):
                        yield (attribute, value)

                    case other:
                        yield attribute + "=" + other

            case other:
                yield other

    raise StopIteration()


def _parse_tag_attribute_name(
    source: Iterator[str],
    /,
) -> tuple[str] | str:
    accumulator: list[str] = []
    while char := next(source, None):
        match char:
            case char if str.isalpha(char) or (
                accumulator and (str.isnumeric(char) or char in "_-")
            ):
                accumulator.append(char)

            case "=" if accumulator:
                return ("".join(accumulator),)

            case other:
                return "".join(accumulator) + other

    return "".join(accumulator)


def _parse_tag_attribute_value(  # noqa: C901, PLR0912
    source: Iterator[str],
    /,
) -> tuple[str] | str:
    # Check first character
    match next(source, ""):
        case '"':
            pass  # continue

        case other:
            return other

    accumulator: list[str] = []
    while char := next(source, None):
        match char:
            case '"':
                return ("".join(accumulator),)

            case "\\":
                # Handle escape sequences
                if (next_char := next(source, None)) is not None:
                    match next_char:
                        case "n":
                            accumulator.append("\n")

                        case "t":
                            accumulator.append("\t")

                        case "r":
                            accumulator.append("\r")

                        case '"' | "\\":
                            accumulator.append(next_char)

                        case _:
                            if next_char == "\n":
                                return '"' + "".join(accumulator) + "\\" + next_char

                            else:
                                accumulator.append("\\" + next_char)

                continue

            case _ if char != "\n":
                accumulator.append(char)

            case _:
                return '"' + "".join(accumulator) + char

    return "".join(accumulator)


class _TagClosing(State):
    name: str
    raw: str
    meta: Meta = META_EMPTY


# look for tag closing and pass through everything else]
# note that we are not allowing spaces before/after tag name
def _parse_tag_closing(  # noqa: C901, PLR0912
    source: Iterator[str],
    /,
    tag: _TagOpening,
    meta: Meta,  # TODO:check
) -> Generator[_TagClosing | str]:
    accumulator: str | None = None  # None is out of tag, empty is a started tag
    closing: bool = False
    while character := next(source, None):
        if accumulator is None:  # check if we started a tag
            match character:
                case "<":  # potential opening
                    accumulator = ""  # prepare accumulator
                    closing = False

                case char:  # skip anything else
                    yield char

        elif closing:
            match character:
                case ">" if accumulator:  # closing
                    if accumulator == tag.name:  # check if it is what we were looking for
                        yield _TagClosing(
                            name=accumulator,
                            raw="</" + accumulator + ">",
                            meta=meta,
                        )
                        return  # end of parsing

                    else:  # otherwise start over again
                        # we are no longer in tag processing, clear out
                        yield "</" + accumulator + ">"  # add opening and closing char
                        accumulator = None

                case char if str.isalnum(char) or char == "_":  # part of the tag name
                    accumulator += char

                case "<":  # potential new tag opening
                    # clear accumulator and start fresh
                    yield "</" + accumulator  # add opening
                    accumulator = ""

                case char:  # anything else
                    # we are no longer in tag processing, clear out
                    yield "</" + accumulator + char  # add opening
                    accumulator = None

        else:
            match character:
                case "/":
                    closing = True

                case char:  # anything else
                    # we are no longer in processing, clear out
                    yield "<" + accumulator + char  # add opening char
                    accumulator = None

    if accumulator is not None:
        if closing:
            yield "</" + accumulator  # add opening char

        else:
            yield "<" + accumulator  # add opening char


def _escaped_attributes(
    attributes: list[tuple[str, str]],
    /,
) -> str:
    return (
        (
            " "
            + " ".join(f'{key}="{_escape_tag_attribute_value(value)}"' for key, value in attributes)
        )
        if attributes
        else ""
    )


def _escape_tag_attribute_value(
    value: str,
    /,
) -> str:
    accumulator: list[str] = []
    for char in value:
        match char:
            case "\n":
                accumulator.append("\\n")

            case "\t":
                accumulator.append("\\t")

            case "\r":
                accumulator.append("\\r")

            case '"':
                accumulator.append('\\"')

            case "\\":
                accumulator.append("\\\\")

            case other:
                accumulator.append(other)

    return "".join(accumulator)


class _CharactersIterator:
    def __init__(
        self,
        text: str,
        /,
    ) -> None:
        self._text: str = text
        self._index: int = 0
        self._size: int = len(text)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> str:
        if self._index >= self._size:
            raise StopIteration()

        try:
            return self._text[self._index]

        finally:
            self._index += 1

    def __bool__(self) -> bool:
        return self._index < self._size
