from typing import Any, TypeGuard

from draive.types.audio import AudioBase64Content, AudioContent, AudioURLContent
from draive.types.images import ImageBase64Content, ImageContent, ImageURLContent
from draive.types.video import VideoBase64Content, VideoContent, VideoURLContent

__all__ = [
    "is_multimodal_content",
    "merge_multimodal_content",
    "multimodal_content_string",
    "has_media",
    "MultimodalContent",
    "MultimodalContentItem",
]

MultimodalContentItem = VideoContent | ImageContent | AudioContent | str
MultimodalContent = tuple[MultimodalContentItem, ...] | MultimodalContentItem


def is_multimodal_content(  # noqa: PLR0911
    candidate: Any,
    /,
) -> TypeGuard[MultimodalContent]:
    match candidate:
        case str():
            return True

        case ImageURLContent():
            return True

        case ImageBase64Content():
            return True

        case AudioURLContent():
            return True

        case AudioBase64Content():
            return True

        case VideoURLContent():
            return True

        case VideoBase64Content():
            return True

        case [*elements] if isinstance(candidate, tuple):
            return all(is_multimodal_content(element) for element in elements)

        case _:
            return False


def has_media(  # noqa: PLR0911
    content: MultimodalContent,
    /,
) -> bool:
    match content:
        case str():
            return False

        case ImageURLContent():
            return True

        case ImageBase64Content():
            return True

        case AudioURLContent():
            return True

        case AudioBase64Content():
            return True

        case VideoURLContent():
            return True

        case VideoBase64Content():
            return True

        case [*elements]:
            return any(has_media(element) for element in elements)


def multimodal_content_string(  # noqa: PLR0911
    content: MultimodalContent,
    /,
) -> str:
    match content:
        case str() as string:
            return string

        case ImageURLContent() as image_url:
            return image_url.image_description or f"[IMAGE]({image_url.image_url})"

        case ImageBase64Content() as image_base64:
            return image_base64.image_description or "[IMAGE]()"

        case AudioURLContent() as audio_url:
            return audio_url.audio_transcription or f"[AUDIO]({audio_url.audio_url})"

        case AudioBase64Content() as audio_base64:
            return audio_base64.audio_transcription or "[AUDIO]()"

        case VideoURLContent() as video_url:
            return video_url.video_transcription or f"[VIDEO]({video_url.video_url})"

        case VideoBase64Content() as video_base64:
            return video_base64.video_transcription or "[VIDEO]()"

        case [*elements]:
            return "\n".join(multimodal_content_string(element) for element in elements)


def merge_multimodal_content(
    *content: MultimodalContent | None,
) -> tuple[MultimodalContentItem, ...]:
    result: list[MultimodalContentItem] = []
    for part in content:
        match part:
            case None:
                continue  # skip none

            case [*parts]:
                result.extend(parts)

            case part:
                result.append(part)

    return tuple(result)
