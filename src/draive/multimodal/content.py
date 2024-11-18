from collections.abc import Iterable, Mapping
from itertools import chain
from typing import Self, final, overload

from haiway import frozenlist

from draive.multimodal.media import MediaContent
from draive.multimodal.text import TextContent
from draive.parameters.model import DataModel

__all__ = [
    "Multimodal",
    "MultimodalContent",
    "MultimodalContentConvertible",
    "MultimodalContentElement",
]


MultimodalContentElement = TextContent | MediaContent | DataModel
MultimodalContentConvertible = str | MultimodalContentElement


@final
class MultimodalContent(DataModel):
    @classmethod
    def of(
        cls,
        *elements: Self | MultimodalContentConvertible,
        meta: Mapping[str, str | float | int | bool | None] | None = None,
    ) -> Self:
        match elements:
            case [MultimodalContent() as content]:
                # if we got just a single content use it as is
                return content  # pyright: ignore[reportReturnType]

            case elements:
                return cls(
                    parts=tuple(
                        _merge_texts(
                            *chain.from_iterable(
                                _extract_parts(
                                    element,
                                    meta=meta,
                                )
                                for element in elements
                            )
                        )
                    ),
                )

    parts: frozenlist[MultimodalContentElement]

    @property
    def has_media(self) -> bool:
        return any(_is_media(part) for part in self.parts)

    @property
    def has_artifacts(self) -> bool:
        return any(_is_artifact(part) for part in self.parts)

    @overload
    def artifacts(self, /) -> frozenlist[DataModel]: ...

    @overload
    def artifacts[Artifact: DataModel](
        self,
        model: type[Artifact],
        /,
    ) -> frozenlist[Artifact]: ...

    def artifacts[Artifact: DataModel](
        self,
        model: type[Artifact] | None = None,
        /,
    ) -> frozenlist[Artifact] | frozenlist[DataModel]:
        if model:
            return tuple(part for part in self.parts if isinstance(part, model))

        else:
            return tuple(part for part in self.parts if _is_artifact(part))

    def excluding_artifacts(self) -> Self:
        return self.__class__(
            parts=tuple(part for part in self.parts if not _is_artifact(part)),
        )

    def as_string(
        self,
        joiner: str | None = None,
        include_data: bool = False,
    ) -> str:
        return (joiner or "\n").join(
            _as_string(
                element,
                include_data=include_data,
            )
            for element in self.parts
        )

    def appending(
        self,
        *parts: MultimodalContentConvertible,
        merge_text: bool = False,
    ) -> Self:
        if not self.parts:
            if merge_text:
                return self.__class__(
                    parts=tuple(_merge_texts(*(_as_content(element) for element in parts))),
                )

            else:
                return self.__class__(
                    parts=tuple(_as_content(element) for element in parts),
                )

        if merge_text:
            # check the last part
            match self.parts[-1]:
                case TextContent() as text:
                    # if it is a text append merge starting with it
                    return self.__class__(
                        parts=(
                            *self.parts[:-1],
                            *_merge_texts(text, *(_as_content(element) for element in parts)),
                        )
                    )

                case _:
                    # otherwise just append merged items
                    return self.__class__(
                        parts=(
                            *self.parts,
                            *_merge_texts(*(_as_content(element) for element in parts)),
                        )
                    )

        else:
            return self.__class__(
                parts=(
                    *self.parts,
                    *(_as_content(element) for element in parts),
                )
            )

    def extending(
        self,
        *other: Self,
        merge_text: bool = False,
    ) -> Self:
        return self.appending(
            *chain.from_iterable(content.parts for content in other),
            merge_text=merge_text,
        )

    def replacing(
        self,
        tag: str,
        /,
        replacement: Self | MultimodalContentConvertible,
        remove_tags: bool = False,
    ) -> Self:
        return _replace_tags(  # pyright: ignore[reportReturnType]
            tag,
            source=self,
            replacement=replacement,
            remove_tags=remove_tags,
        )

    def __bool__(self) -> bool:
        return bool(self.parts) and any(self.parts)

    def __str__(self) -> str:
        return self.as_string()


Multimodal = MultimodalContent | MultimodalContentConvertible


def _extract_parts(  # noqa: PLR0911
    element: Multimodal,
    /,
    meta: Mapping[str, str | float | int | bool | None] | None = None,
) -> frozenlist[MultimodalContentElement]:
    match element:
        case MultimodalContent() as content:
            if not content:
                return ()

            elif meta:
                return tuple(
                    _update_meta(
                        meta,
                        element=part,
                    )
                    for part in content.parts
                )

            else:
                return content.parts

        case str() as text:
            if not text:
                return ()

            else:
                return (
                    TextContent(
                        text=text,
                        meta=meta,
                    ),
                )

        case element:
            if not element:
                return ()

            elif meta:
                return (
                    _update_meta(
                        meta,
                        element=element,
                    ),
                )

            else:
                return (element,)


def _as_content(
    element: MultimodalContentConvertible,
    /,
) -> MultimodalContentElement:
    match element:
        case str() as text:
            return TextContent(text=text)

        case element:
            return element


def _is_media(
    element: MultimodalContentElement,
) -> bool:
    return isinstance(
        element,
        MediaContent,
    )


def _is_artifact(
    element: MultimodalContentElement,
) -> bool:
    return not isinstance(
        element,
        TextContent | MediaContent,
    )


def _as_string(
    element: MultimodalContentElement,
    /,
    *,
    include_data: bool,
) -> str:
    match element:
        case TextContent() as text:
            return text.text

        case MediaContent() as media:
            return media.as_string(include_data=include_data)

        case DataModel() as model:
            return str(model)


def _merge_texts(
    *elements: MultimodalContentElement,
) -> Iterable[MultimodalContentElement]:
    if len(elements) <= 1:
        return elements

    result: list[MultimodalContentElement] = []
    last_text_element: TextContent | None = None
    for element in elements:
        match element:
            case TextContent() as text:
                # do not merge texts with different metadata
                if last_text := last_text_element:
                    if last_text.meta == text.meta:
                        last_text_element = TextContent(
                            text=last_text.text + text.text,
                            meta=text.meta,
                        )

                    else:
                        result.append(last_text)
                        last_text_element = text

                else:
                    last_text_element = text

            case other:
                if last_text := last_text_element:
                    result.append(last_text)
                    last_text_element = None

                result.append(other)

    if last_text := last_text_element:
        result.append(last_text)

    return result


def _update_meta(
    meta: Mapping[str, str | float | int | bool | None],
    /,
    element: MultimodalContentElement,
) -> MultimodalContentElement:
    match element:
        case (TextContent() | MediaContent()) as content:
            if current_meta := content.meta:
                return content.updated(meta={**current_meta, **meta})

            else:
                return content.updated(meta=meta)

        case DataModel() as model:
            return model


def _extract_replacement_content(
    replacement: Multimodal,
    /,
) -> tuple[MultimodalContentElement, ...]:
    match replacement:
        case str() as text:
            return (TextContent(text=text),)

        case MultimodalContent() as multimodal:
            return multimodal.parts

        case element:
            return (element,)


# TODO: this function requires serious refactoring / rewrite
def _replace_tags(  # noqa: C901, PLR0912, PLR0915
    tag: str,
    /,
    source: MultimodalContent,
    replacement: Multimodal,
    remove_tags: bool,
) -> MultimodalContent:
    replacement_content: tuple[MultimodalContentElement, ...] = _extract_replacement_content(
        replacement
    )

    tag_opening_prefix: str = f"<{tag}"
    tag_closing_prefix: str = f"</{tag}"

    tag_accumulator: str = ""  # tag detection buffer
    tag_parts: list[TextContent] | None = None  # current tag detection parts with meta
    tag_parts_len: int = 0  # parts text length
    opening_tag_parts: list[TextContent] | None = None  # opening tag parts with meta
    tag_content_parts: list[MultimodalContentElement] | None = None  # tag content after opening
    result_parts: list[MultimodalContentElement] = []  # final result

    for part in source.parts:
        match part:  # iterate over content parts
            case TextContent() as text:
                text_part_accumulator: str = ""  # current text part accumulator
                for char in text.text:  # iterate over text characters
                    if tag_accumulator:  # check the inside of potential tag
                        # we have potential tag closing
                        if char == ">":
                            # we are no longer in tag since we are not handling
                            # the '>' character within xml tag properties even inside a string

                            # we have found a closing tag
                            if tag_accumulator == tag_closing_prefix:
                                if remove_tags:
                                    # add tag content replacement directly to the result
                                    # skip tag opening (if any) and closing
                                    result_parts.extend(replacement_content)

                                else:
                                    # add or create tag opening
                                    if opening_tag_parts:
                                        result_parts.extend(opening_tag_parts)

                                    else:
                                        result_parts.append(
                                            TextContent(
                                                # don't miss last character
                                                text=tag_opening_prefix + ">",
                                                meta=None,  # we are creating this tag without meta
                                            )
                                        )

                                    # add tag content replacement directly to the result
                                    result_parts.extend(replacement_content)

                                    # check for multi-part tag
                                    if tag_parts:
                                        # then add closing tag to the result
                                        result_parts.extend(tag_parts)
                                        # finally add the closing tag reminder with local meta
                                        result_parts.append(
                                            TextContent(
                                                # don't miss last character
                                                text=tag_accumulator[tag_parts_len:] + ">",
                                                meta=text.meta,
                                            )
                                        )

                                    else:
                                        # or add full closing tag to the result
                                        result_parts.append(
                                            TextContent(
                                                # don't miss last character
                                                text=tag_accumulator + ">",
                                                meta=text.meta,
                                            )
                                        )

                                # we have completed tag detection
                                tag_accumulator = ""
                                # we are no longer in content
                                tag_content_parts = None
                                # we are no longer in tag
                                opening_tag_parts = None
                                # we have consumed tag parts
                                tag_parts = None
                                tag_parts_len = 0

                            # we have found opening tag
                            elif tag_accumulator == tag_opening_prefix:
                                # use previous opening candidate in the result
                                if opening_tag_parts:
                                    result_parts.extend(opening_tag_parts)

                                # use previous content candidate in the result
                                if tag_content_parts:
                                    result_parts.extend(tag_content_parts)

                                # add opening tag to the opening_tag_parts
                                if tag_parts:
                                    opening_tag_parts = [
                                        *tag_parts,
                                        TextContent(
                                            # don't miss last character
                                            text=tag_accumulator[tag_parts_len:] + ">",
                                            meta=text.meta,
                                        ),
                                    ]

                                else:  # otherwise use what we have
                                    opening_tag_parts = [
                                        TextContent(
                                            # don't miss last character
                                            text=tag_accumulator + ">",
                                            meta=text.meta,
                                        )
                                    ]

                                # we have completed tag detection
                                tag_accumulator = ""
                                # we are in a new content now
                                tag_content_parts = []
                                # we have consumed tag parts
                                tag_parts = None
                                tag_parts_len = 0

                            else:  # we fond nothing - add it to current parts
                                if tag_parts:
                                    if tag_content_parts is None:
                                        result_parts.extend(tag_parts)

                                    else:
                                        tag_content_parts.extend(tag_parts)

                                # turn tag accumulator into current text part
                                text_part_accumulator += tag_accumulator[tag_parts_len:] + char

                                # we have completed tag detection
                                tag_accumulator = ""
                                # do not change in content state or opening tag here
                                # we have consumed tag parts
                                tag_parts = None
                                tag_parts_len = 0

                        # we have potential new tag opening
                        elif char == "<":
                            # turn tag accumulator into current text part
                            text_part_accumulator += tag_accumulator
                            # handle current text part to the current content if needed
                            if tag_content_parts is None:
                                if tag_parts:
                                    result_parts.extend(tag_parts)

                                if text_part_accumulator:
                                    result_parts.append(
                                        TextContent(
                                            text=text_part_accumulator,
                                            meta=text.meta,
                                        )
                                    )

                            else:
                                if tag_content_parts:
                                    tag_content_parts.extend(tag_content_parts)

                                if tag_parts:
                                    tag_content_parts.extend(tag_parts)

                                if text_part_accumulator:
                                    tag_content_parts.append(
                                        TextContent(
                                            text=text_part_accumulator,
                                            meta=text.meta,
                                        )
                                    )

                            text_part_accumulator = ""

                            # we have started tag detection
                            tag_accumulator = char
                            # do not change in content state or tag opening here
                            # we have consumed tag parts
                            tag_parts = None
                            tag_parts_len = 0

                        else:  # treat it as a part of tag otherwise
                            tag_accumulator += char

                    elif char == "<":  # check potential new tag opening
                        # handle current text part to the current content if needed
                        if tag_content_parts is None:
                            if tag_parts:
                                result_parts.extend(tag_parts)

                            if text_part_accumulator:
                                result_parts.append(
                                    TextContent(
                                        text=text_part_accumulator,
                                        meta=text.meta,
                                    )
                                )

                        else:
                            if tag_parts:
                                tag_content_parts.extend(tag_parts)

                            if text_part_accumulator:
                                tag_content_parts.append(
                                    TextContent(
                                        text=text_part_accumulator,
                                        meta=text.meta,
                                    )
                                )

                        text_part_accumulator = ""

                        # we have started tag detection
                        tag_accumulator = char
                        # do not change in content state or tag opening here
                        # we have consumed tag parts
                        tag_parts = None
                        tag_parts_len = 0

                    else:  # use what we got otherwise
                        text_part_accumulator += char

                # check for the text content leftovers
                if text_part_accumulator:
                    leftover_content = TextContent(
                        text=text_part_accumulator,
                        meta=text.meta,
                    )
                    if tag_content_parts is None:
                        result_parts.append(leftover_content)

                    else:
                        tag_content_parts.append(leftover_content)

                # check the tag detection accumulator leftovers
                if leftover := tag_accumulator[tag_parts_len:]:
                    leftover_content = TextContent(
                        text=leftover,
                        meta=text.meta,
                    )
                    # and add it to the dedicated leftovers cache
                    if tag_parts is None:
                        tag_parts = [leftover_content]

                    else:
                        tag_parts.append(leftover_content)

                    # ensure proper len counting
                    tag_parts_len += len(leftover)

            case model:  # media or custom model
                if tag_parts:  # cleanup tag detection if needed
                    # add leftovers to the appropriate container
                    if tag_content_parts is None:
                        result_parts.extend(tag_parts)

                    else:
                        tag_content_parts.extend(tag_parts)

                    # clear the tag detection accumulator
                    tag_accumulator = ""
                    # and ensure we are no longer in tag detection
                    tag_parts = None
                    tag_parts_len = 0

                # add any non-text content to the appropriate container
                if tag_content_parts is None:
                    result_parts.append(model)

                else:
                    tag_content_parts.append(model)

    # add all remaining parts to the result
    if opening_tag_parts:
        result_parts.extend(opening_tag_parts)

    if tag_content_parts:
        result_parts.extend(tag_content_parts)

    if tag_parts:
        result_parts.extend(tag_parts)

    return MultimodalContent.of(*result_parts)
