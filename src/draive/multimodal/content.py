from collections.abc import Sequence
from typing import ClassVar, Self, cast, final, overload

from draive.commons import Meta, MetaValues
from draive.multimodal.media import MediaContent, MediaData, MediaKind, MediaReference
from draive.multimodal.meta import MetaContent
from draive.multimodal.text import TextContent
from draive.parameters import DataModel

__all__ = (
    "Multimodal",
    "MultimodalContent",
    "MultimodalContentConvertible",
    "MultimodalContentElement",
)


MultimodalContentElement = TextContent | MediaContent | MetaContent | DataModel
MultimodalContentConvertible = str | MultimodalContentElement


@final  # TODO: optimize performance
class MultimodalContent(DataModel):
    empty: ClassVar[Self]  # defined after the class

    @classmethod
    def of(
        cls,
        *elements: Self | MultimodalContentConvertible,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        if not elements:
            return cls.empty

        if len(elements) == 1 and isinstance(elements[0], MultimodalContent) and meta is None:
            return cast(Self, elements[0])

        meta_obj: Meta | None = Meta.of(meta) if meta is not None else None

        all_parts: list[MultimodalContentElement] = []
        for element in elements:
            all_parts.extend(
                _extract_parts(
                    element,
                    meta=meta_obj,
                )
            )

        return cls(parts=_merge_texts(*all_parts))

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

    def images(self) -> Self:
        return self.__class__(
            parts=tuple(
                part
                for part in self.parts
                if isinstance(part, MediaContent) and part.kind == "image"
            )
        )

    def audio(self) -> Self:
        return self.__class__(
            parts=tuple(
                part
                for part in self.parts
                if isinstance(part, MediaContent) and part.kind == "audio"
            )
        )

    def video(self) -> Self:
        return self.__class__(
            parts=tuple(
                part
                for part in self.parts
                if isinstance(part, MediaContent) and part.kind == "video"
            )
        )

    def is_media(
        self,
        media: MediaKind | None = None,
    ) -> bool:
        if media is None:
            return all(isinstance(part, MediaContent) for part in self.parts)

        else:
            return all(isinstance(part, MediaContent) and part.kind == media for part in self.parts)

    def without_media(self) -> Self:
        return self.__class__(
            parts=tuple(part for part in self.parts if not isinstance(part, MediaContent)),
        )

    def is_artifact(self) -> bool:
        return all(_is_artifact(part) for part in self.parts)

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

    def is_meta(self) -> bool:
        return all(isinstance(part, MetaContent) for part in self.parts)

    def meta(
        self,
        *,
        category: str | None = None,
    ) -> Sequence[MetaContent]:
        if category is not None:
            return tuple(
                part
                for part in self.parts
                if isinstance(part, MetaContent) and part.category == category
            )

        else:
            return tuple(part for part in self.parts if isinstance(part, MetaContent))

    def without_meta(self) -> Self:
        return self.__class__(
            parts=tuple(part for part in self.parts if not isinstance(part, MetaContent)),
        )

    def text(self) -> Self:
        return self.__class__(
            parts=tuple(part for part in self.parts if isinstance(part, TextContent)),
        )

    def to_str(
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
        *parts: Self | MultimodalContentConvertible,
    ) -> Self:
        if not parts:
            return self

        all_new_parts: list[MultimodalContentElement] = []
        for part in parts:
            all_new_parts.extend(_extract_parts(part))

        if len(self.parts) == 0:
            return self.__class__(
                parts=tuple(_merge_texts(*all_new_parts)),
            )

        if isinstance(self.parts[-1], TextContent):
            return self.__class__(
                parts=(
                    *self.parts[:-1],
                    *_merge_texts(self.parts[-1], *all_new_parts),
                )
            )

        else:
            return self.__class__(
                parts=(
                    *self.parts,
                    *_merge_texts(*all_new_parts),
                )
            )

    def extended_by(
        self,
        *other: Self,
    ) -> Self:
        # Extract all parts directly
        all_other_parts: list[MultimodalContentElement] = []
        for content in other:
            all_other_parts.extend(content.parts)

        return self.appending(*all_other_parts)

    def __bool__(self) -> bool:
        return bool(self.parts) and any(self.parts)

    def __str__(self) -> str:
        return self.to_str()


MultimodalContent.empty = MultimodalContent(parts=())
Multimodal = MultimodalContent | MultimodalContentConvertible


def _extract_parts(  # noqa: PLR0911
    element: Multimodal,
    /,
    meta: Meta | None = None,
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
                        meta=Meta.of(meta),
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

        case MediaData() as media_data:
            return media_data.to_str(include_data=include_data)

        case MediaReference() as media_reference:
            return media_reference.to_str()

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
    accumulated_texts: list[str] = []
    current_meta: Meta | None = None

    for element in elements:
        if isinstance(element, TextContent):
            if current_meta is None:
                current_meta = element.meta
                accumulated_texts.append(element.text)

            elif current_meta == element.meta:
                accumulated_texts.append(element.text)

            else:
                if accumulated_texts and current_meta is not None:
                    result.append(
                        TextContent(
                            text="".join(accumulated_texts),
                            meta=current_meta,
                        )
                    )

                current_meta = element.meta
                accumulated_texts = [element.text]

        else:
            if accumulated_texts and current_meta is not None:
                result.append(
                    TextContent(
                        text="".join(accumulated_texts),
                        meta=current_meta,
                    )
                )
                accumulated_texts = []
                current_meta = None

            result.append(element)

    if accumulated_texts and current_meta is not None:
        result.append(
            TextContent(
                text="".join(accumulated_texts),
                meta=current_meta,
            )
        )

    return result


def _update_meta(
    meta: Meta,
    /,
    element: MultimodalContentElement,
) -> MultimodalContentElement:
    match element:
        case (TextContent() | MediaData() | MediaReference() | MetaContent()) as content:
            if current_meta := content.meta:
                return content.updated(meta={**current_meta, **meta})

            else:
                return content.updated(meta=meta)

        case DataModel() as model:
            return model
