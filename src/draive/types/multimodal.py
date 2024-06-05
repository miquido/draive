from itertools import chain
from typing import Self, final

from draive.parameters.model import DataModel
from draive.types.audio import AudioBase64Content, AudioContent, AudioDataContent, AudioURLContent
from draive.types.frozenlist import frozenlist
from draive.types.images import ImageBase64Content, ImageContent, ImageDataContent, ImageURLContent
from draive.types.video import VideoBase64Content, VideoContent, VideoDataContent, VideoURLContent

__all__ = [
    "MultimodalContent",
    "MultimodalContentElement",
]

MultimodalContentElement = VideoContent | ImageContent | AudioContent | str


@final
class MultimodalContent(DataModel):
    @classmethod
    def of(
        cls,
        *elements: Self | MultimodalContentElement,
    ) -> Self:
        match elements:
            case [MultimodalContent() as content]:
                return content

            case elements:
                return cls(
                    elements=tuple(chain.from_iterable(_extract(element) for element in elements)),
                )

    elements: frozenlist[MultimodalContentElement]

    @property
    def has_media(self) -> bool:
        return any(_is_media(element) for element in self.elements)

    def as_string(
        self,
        joiner: str | None = None,
    ) -> str:
        return (joiner or "\n").join(_as_string(element) for element in self.elements)

    def appending(
        self,
        *elements: MultimodalContentElement,
    ) -> Self:
        return self.__class__(
            elements=(
                *self.elements,
                *elements,
            )
        )

    def extending(
        self,
        *other: Self,
    ) -> Self:
        return self.__class__(
            elements=(
                *self.elements,
                *(element for content in other for element in content.elements),
            )
        )

    def joining_texts(
        self,
        joiner: str | None = None,
    ) -> Self:
        joined_elements: list[MultimodalContentElement] = []
        current_text: str | None = None
        for element in self.elements:
            match element:
                case str() as string:
                    if current_text:
                        current_text = (joiner or "\n").join((current_text, string))

                    else:
                        current_text = string

                case other:
                    if current_text:
                        joined_elements.append(current_text)
                        current_text = None

                    joined_elements.append(other)

        return self.__class__(
            elements=tuple(joined_elements),
        )

    def __bool__(self) -> bool:
        return bool(self.elements) and any(self.elements)


def _extract(
    element: MultimodalContent | MultimodalContentElement,
    /,
) -> frozenlist[MultimodalContentElement]:
    match element:
        case MultimodalContent() as content:
            return content.elements

        case element:
            return (element,)


def _is_media(
    element: MultimodalContentElement,
) -> bool:
    match element:
        case str():
            return False

        case _:
            return True


def _as_string(  # noqa: PLR0911, C901
    element: MultimodalContentElement,
) -> str:
    match element:
        case str() as string:
            return string

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
