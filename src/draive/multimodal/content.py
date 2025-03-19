from collections.abc import Sequence
from itertools import chain
from typing import ClassVar, Self, final, overload

from draive.commons import META_EMPTY, Meta
from draive.multimodal.media import MediaContent, MediaKind
from draive.multimodal.meta import MetaContent
from draive.multimodal.text import TextContent
from draive.parameters import DataModel

__all__ = [
    "Multimodal",
    "MultimodalContent",
    "MultimodalContentConvertible",
    "MultimodalContentElement",
]


MultimodalContentElement = TextContent | MediaContent | MetaContent | DataModel
MultimodalContentConvertible = str | MultimodalContentElement


@final
class MultimodalContent(DataModel):
    empty: ClassVar[Self]  # defined after the class

    @classmethod
    def of(
        cls,
        *elements: Self | MultimodalContentConvertible,
        meta: Meta | None = None,
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
                                    meta=meta if meta is not None else META_EMPTY,
                                )
                                for element in elements
                            )
                        )
                    ),
                )

    parts: Sequence[MultimodalContentElement]

    @property
    def has_media(self) -> bool:
        return any(_is_media(part) for part in self.parts)

    def media(
        self,
        media: MediaKind | None = None,
    ) -> Sequence[MediaContent]:
        if media is None:
            return tuple(part for part in self.parts if isinstance(part, MediaContent))

        else:
            return tuple(
                part for part in self.parts if isinstance(part, MediaContent) and part.kind == media
            )

    def without_media(self) -> Self:
        return self.__class__(
            parts=tuple(part for part in self.parts if not isinstance(part, MediaContent)),
        )

    @property
    def has_artifacts(self) -> bool:
        return any(_is_artifact(part) for part in self.parts)

    @overload
    def artifacts(
        self,
        /,
    ) -> Sequence[DataModel]: ...

    @overload
    def artifacts[Artifact: DataModel](
        self,
        model: type[Artifact],
        /,
    ) -> Sequence[Artifact]: ...

    def artifacts[Artifact: DataModel](
        self,
        model: type[Artifact] | None = None,
        /,
    ) -> Sequence[Artifact] | Sequence[DataModel]:
        if model is None:
            return tuple(part for part in self.parts if _is_artifact(part))

        else:
            return tuple(part for part in self.parts if isinstance(part, model))

    def without_artifacts(self) -> Self:
        return self.__class__(
            parts=tuple(part for part in self.parts if not _is_artifact(part)),
        )

    def meta(self) -> Sequence[MetaContent]:
        return tuple(part for part in self.parts if isinstance(part, MetaContent))

    def without_meta(self) -> Self:
        return self.__class__(
            parts=tuple(part for part in self.parts if not isinstance(part, MetaContent)),
        )

    def as_string(
        self,
        joiner: str | None = None,
        include_data: bool = False,
    ) -> str:
        return (joiner if joiner is not None else "").join(
            _as_string(
                element,
                include_data=include_data,
            )
            for element in self.parts
        )

    def appending(
        self,
        *parts: MultimodalContentConvertible,
    ) -> Self:
        assert not any(isinstance(part, MultimodalContent) for part in parts)  # nosec: B101
        if len(self.parts) == 0:
            return self.__class__(
                parts=tuple(_merge_texts(*(_as_content(element) for element in parts))),
            )

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

    def extending(
        self,
        *other: Self,
    ) -> Self:
        return self.appending(*chain.from_iterable(content.parts for content in other))

    def __bool__(self) -> bool:
        return bool(self.parts) and any(self.parts)

    def __str__(self) -> str:
        return self.as_string()


MultimodalContent.empty = MultimodalContent(parts=())
Multimodal = MultimodalContent | MultimodalContentConvertible


def _extract_parts(  # noqa: PLR0911
    element: Multimodal,
    /,
    meta: Meta,
) -> Sequence[MultimodalContentElement]:
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
        TextContent | MediaContent | MetaContent,
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

        case MetaContent() as meta:
            return (  # perhaps, we could use meta values within xml tag?
                (
                    f"<{meta.category}>"
                    f"{_as_string(meta.content, include_data=include_data)}"
                    f"</{meta.category}>"
                )
                if meta.content is not None
                else f"</{meta.category}>"
            )

        case DataModel() as model:
            return str(model)


def _merge_texts(
    *elements: MultimodalContentElement,
) -> Sequence[MultimodalContentElement]:
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
    meta: Meta,
    /,
    element: MultimodalContentElement,
) -> MultimodalContentElement:
    match element:
        case (TextContent() | MediaContent() | MetaContent()) as content:
            if current_meta := content.meta:
                return content.updated(meta={**current_meta, **meta})

            else:
                return content.updated(meta=meta)

        case DataModel() as model:
            return model
