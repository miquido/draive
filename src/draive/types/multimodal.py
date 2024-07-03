from collections.abc import Iterable, Iterator
from itertools import chain
from typing import Self, final, overload

from draive.parameters.model import DataModel
from draive.types.audio import AudioBase64Content, AudioContent, AudioDataContent, AudioURLContent
from draive.types.frozenlist import frozenlist
from draive.types.image import ImageBase64Content, ImageContent, ImageDataContent, ImageURLContent
from draive.types.text import TextContent
from draive.types.video import VideoBase64Content, VideoContent, VideoDataContent, VideoURLContent

__all__ = [
    "MultimodalContent",
    "MultimodalContentElement",
    "MultimodalContentConvertible",
]

MultimodalContentElement = TextContent | ImageContent | AudioContent | VideoContent | DataModel
MultimodalContentConvertible = str | MultimodalContentElement


@final
class MultimodalContent(DataModel):
    @classmethod
    def of(
        cls,
        *elements: Self | MultimodalContentConvertible,
        merge_text: bool = False,
    ) -> Self:
        match elements:
            case [MultimodalContent() as content] if not merge_text:
                return content  # pyright: ignore[reportReturnType]

            case elements:
                if merge_text:
                    return cls(
                        parts=tuple(
                            _merge_texts(
                                *chain.from_iterable(
                                    _extract_parts(element) for element in elements
                                )
                            )
                        ),
                    )

                else:
                    return cls(
                        parts=tuple(
                            chain.from_iterable(_extract_parts(element) for element in elements)
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
    ) -> str:
        return (joiner or "\n").join(_as_string(element) for element in self.parts)

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

    def __bool__(self) -> bool:
        return bool(self.parts) and any(self.parts)


def _extract_parts(
    element: MultimodalContent | MultimodalContentConvertible,
    /,
) -> frozenlist[MultimodalContentElement]:
    match element:
        case MultimodalContent() as content:
            return content.parts

        case str() as text:
            return (TextContent(text=text),)

        case element:
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
    match element:
        case TextContent():
            return False

        case _:
            return True


def _is_artifact(  # noqa: C901, PLR0911
    element: MultimodalContentElement,
) -> bool:
    match element:
        case TextContent():
            return False

        case ImageURLContent():
            return False

        case ImageBase64Content():
            return False

        case ImageDataContent():
            return False

        case AudioURLContent():
            return False

        case AudioBase64Content():
            return False

        case AudioDataContent():
            return False

        case VideoURLContent():
            return False

        case VideoBase64Content():
            return False

        case VideoDataContent():
            return False

        case _:
            return True


def _as_string(  # noqa: PLR0911, C901
    element: MultimodalContentElement,
) -> str:
    match element:
        case TextContent() as text:
            return text.text

        case ImageURLContent() as image_url:
            return f"![{image_url.image_description or 'IMAGE'}]({image_url.image_url})"

        case ImageBase64Content() as image_base64:
            # we might want to use base64 content directly, but it would make a lot of tokens...
            return f"![{image_base64.image_description or 'IMAGE'}]()"

        case ImageDataContent() as image_data:
            # we might want to convert to base64 content, but it would make a lot of tokens...
            return f"![{image_data.image_description or 'IMAGE'}]()"

        case AudioURLContent() as audio_url:
            return f"![{audio_url.audio_transcription or 'AUDIO'}]({audio_url.audio_url})"

        case AudioBase64Content() as audio_base64:
            # we might want to use base64 content directly, but it would make a lot of tokens...
            return f"![{audio_base64.audio_transcription or 'AUDIO'}]()"

        case AudioDataContent() as audio_data:
            # we might want to convert to base64 content, but it would make a lot of tokens...
            return f"![{audio_data.audio_transcription or 'AUDIO'}]()"

        case VideoURLContent() as video_url:
            return f"![{video_url.video_transcription or 'VIDEO'}]({video_url.video_url})"

        case VideoBase64Content() as video_base64:
            # we might want to use base64 content directly, but it would make a lot of tokens...
            return f"![{video_base64.video_transcription or 'VIDEO'}]()"

        case VideoDataContent() as video_data:
            # we might want to convert to base64 content, but it would make a lot of tokens...
            return f"![{video_data.video_transcription or 'VIDEO'}]()"

        case DataModel() as model:
            return str(model)


def _merge_texts(
    *elements: MultimodalContentElement,
) -> Iterable[MultimodalContentElement]:
    if len(elements) <= 1:
        return elements

    result: list[MultimodalContentElement] = []
    iterator: Iterator[MultimodalContentElement] = iter(elements)
    last_text_element: TextContent | None = None
    while element := next(iterator, None):
        match element:
            case TextContent() as text:
                if last_text := last_text_element:
                    last_text_element = TextContent(text=last_text.text + text.text)

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
