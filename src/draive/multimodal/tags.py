from collections.abc import Generator, Iterable, Iterator, Mapping
from typing import Final, Self, cast

from haiway import State

from draive.multimodal.content import Multimodal, MultimodalContent, MultimodalContentElement
from draive.multimodal.text import TextContent

__all__ = [
    "MultimodalTagElement",
]


# TODO: add streaming support with content input as async iterator
class MultimodalTagElement(State):
    name: str
    attributes: Mapping[str, str] | None = None
    content: MultimodalContent

    @classmethod
    def parse_first(
        cls,
        content: Multimodal,
        /,
        tag: str | None = None,
    ) -> Self | None:
        return next(
            cls.parse(
                content,
                tag=tag,
            ),
            None,
        )

    @classmethod
    def parse(  # noqa: C901, PLR0912
        cls,
        content: Multimodal,
        /,
        tag: str | None = None,
    ) -> Generator[Self]:
        # TODO: add support for split meta tag opening and closing?
        current_tag: _MultimodalTag | None = None
        current_tag_content: list[MultimodalContentElement] = []
        for part in MultimodalContent.of(content).parts:
            match part:
                case TextContent() as text_content:
                    text_iterator: _ParserIterator[str] = _ParserIterator(text_content.text)
                    text_accumulator: str = ""  # accumulator for any parts that are not tags
                    while text_iterator:  # keep working until end of text part
                        if current_tag is None:  # find the tag opening first
                            for element in _parse_tag_opening(
                                text_iterator,
                                meta=text_content.meta,
                            ):
                                match element:
                                    case _MultimodalTag() as opening:
                                        # begin new tag
                                        current_tag = opening
                                        # clear content and accumulator
                                        current_tag_content = []
                                        text_accumulator = ""
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
                                    case _MultimodalTag():
                                        # check if it is the tag we are looking for
                                        if not tag or current_tag.name == tag:
                                            if text_accumulator:
                                                current_tag_content.append(
                                                    TextContent(
                                                        text=text_accumulator,
                                                        meta=text_content.meta,
                                                    )
                                                )

                                            yield cls(
                                                name=current_tag.name,
                                                attributes=current_tag.attributes,
                                                content=MultimodalContent.of(*current_tag_content),
                                            )

                                        # skip unwanted tags and clear out
                                        current_tag = None
                                        current_tag_content = []
                                        text_accumulator = ""
                                        break  # and break the loop - we have closing

                                    case text:
                                        text_accumulator += text

                        # continue until end of part text

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
        content: Multimodal,
        /,
        tag: str,
        replacement: Multimodal,
        strip_tags: bool = False,
        replace_all: bool = False,
    ) -> MultimodalContent:
        # TODO: add support for tag id attribute or custom matching?
        # TODO: add support for split meta tag opening and closing?
        replacement_content = MultimodalContent.of(replacement).parts
        result_content: list[MultimodalContentElement] = []
        current_tag: _MultimodalTag | None = None
        current_tag_content: list[MultimodalContentElement] = []
        replaced: bool = False
        for part in MultimodalContent.of(content).parts:
            match part:
                case TextContent() as text_content:
                    text_iterator: _ParserIterator[str] = _ParserIterator(text_content.text)
                    text_accumulator: str = ""  # accumulator for any parts that are not tags
                    while text_iterator:  # keep working until end of text part
                        if current_tag is None:  # find the tag opening first
                            for element in _parse_tag_opening(
                                text_iterator,
                                meta=text_content.meta,
                            ):
                                match element:
                                    case _MultimodalTag() as opening:
                                        # include all content up to this point
                                        if text_accumulator:
                                            result_content.append(
                                                TextContent(
                                                    text=text_accumulator,
                                                    meta=text_content.meta,
                                                )
                                            )

                                        # begin new tag
                                        current_tag = opening
                                        # clear content and accumulator
                                        current_tag_content = []
                                        text_accumulator = ""
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
                                    case _MultimodalTag() as closing:
                                        # check if it is the tag we are looking for
                                        if (not tag or current_tag.name == tag) and (
                                            replace_all or not replaced
                                        ):
                                            replaced = True

                                            if strip_tags:
                                                result_content.extend(replacement_content)

                                            else:
                                                result_content.extend(
                                                    (
                                                        current_tag.raw,
                                                        *replacement_content,
                                                        closing.raw,
                                                    ),
                                                )

                                        else:
                                            if text_accumulator:
                                                current_tag_content.append(
                                                    TextContent(
                                                        text=text_accumulator,
                                                        meta=text_content.meta,
                                                    )
                                                )

                                            result_content.extend(
                                                (
                                                    current_tag.raw,
                                                    *current_tag_content,
                                                    closing.raw,
                                                ),
                                            )

                                        # skip unwanted tags and clear out
                                        current_tag = None
                                        current_tag_content = []
                                        text_accumulator = ""
                                        break  # and break the loop - we have closing

                                    case text:
                                        text_accumulator += text

                        # continue until end of part text

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
                    current_tag.raw,
                    MultimodalContent.of(*current_tag_content),
                ),
            )
        return MultimodalContent.of(*result_content)


class _MultimodalTag(State):
    name: str
    attributes: Mapping[str, str] | None
    raw: TextContent
    closed: bool


# look for tag opening and pass through everything else
def _parse_tag_opening(
    source: Iterator[str],
    /,
    meta: Mapping[str, str | float | int | bool | None] | None,
) -> Generator[_MultimodalTag | str]:
    accumulator: str | None = None  # None is out of tag, empty is a started tag
    while character := next(source, None):
        if accumulator is None:  # check if we started a tag
            match character:
                case "<":  # potential tag opening
                    accumulator = ""  # prepare accumulator

                case char:  # skip anything else
                    yield char

        else:
            match character:
                case ">":  # tag closing
                    # TODO: add closed tag support
                    yield _MultimodalTag(
                        name=accumulator,
                        attributes=None,  # TODO: add attributes parsing support
                        raw=TextContent(
                            text="<" + accumulator + ">",
                            meta=meta,
                        ),
                        closed=False,
                    )
                    return  # end of parsing

                case char if str.isalnum(char):  # part of the tag name
                    accumulator += char

                case "<":  # potential new tag opening
                    # clear accumulator and start fresh
                    yield "<" + accumulator  # add opening char
                    accumulator = ""

                case char:  # anything else
                    # we are no longer in tag processing, clear out
                    yield "<" + accumulator + char  # add opening char
                    accumulator = None

    if accumulator is not None:
        yield "<" + accumulator  # add opening char


# look for tag closing and pass through everything else
def _parse_tag_closing(  # noqa: C901, PLR0912
    source: Iterator[str],
    /,
    tag: _MultimodalTag,
    meta: Mapping[str, str | float | int | bool | None] | None,
) -> Generator[_MultimodalTag | str]:
    accumulator: str | None = None  # None is out of tag, empty is a started tag
    closing: bool = False
    while character := next(source, None):
        if accumulator is None:  # check if we started a tag
            match character:
                case "<":  # potential tag opening
                    accumulator = ""  # prepare accumulator
                    closing = False

                case char:  # skip anything else
                    yield char

        elif closing:
            match character:
                case ">":  # tag closing
                    if accumulator == tag.name:  # check if it is what we were looking for
                        yield _MultimodalTag(
                            name=accumulator,
                            attributes=tag.attributes,
                            raw=TextContent(
                                text="</" + accumulator + ">",
                                meta=meta,
                            ),
                            closed=True,
                        )
                        return  # end of parsing

                    else:  # otherwise start over again
                        # we are no longer in tag processing, clear out
                        yield "</" + accumulator + ">"  # add opening and closing char
                        accumulator = None

                case char if str.isalnum(char):  # part of the tag name
                    accumulator += char

                case "<":  # potential new tag opening
                    # clear accumulator and start fresh
                    yield "</" + accumulator  # add opening char
                    accumulator = ""

                case char:  # anything else
                    # we are no longer in tag processing, clear out
                    yield "</" + accumulator + char  # add opening char
                    accumulator = None

        else:
            match character:
                case "/":
                    closing = True

                case char:  # anything else
                    # we are no longer in tag processing, clear out
                    yield "<" + accumulator + char  # add opening char
                    accumulator = None

    if accumulator is not None:
        if closing:
            yield "</" + accumulator  # add opening char

        else:
            yield "<" + accumulator  # add opening char


_SENTINEL: Final[object] = object()


class _ParserIterator[T]:
    def __init__(
        self,
        iterable: Iterable[T],
    ) -> None:
        self._iterator: Iterator[T] = iter(iterable)
        self._peek: T | object = _SENTINEL

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        if self._peek is _SENTINEL:
            return next(self._iterator)

        peek: T = cast(T, self._peek)
        self._peek = _SENTINEL
        return peek

    def __bool__(self) -> bool:
        if self._peek is _SENTINEL:
            try:
                self._peek = next(self._iterator)

            except StopIteration:
                return False

        return True
