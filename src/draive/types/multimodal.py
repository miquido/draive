from draive.types.audio import AudioContent
from draive.types.images import ImageContent
from draive.types.video import VideoContent

__all__ = [
    "merge_multimodal_content",
    "MultimodalContent",
    "MultimodalContentItem",
]

MultimodalContentItem = VideoContent | ImageContent | AudioContent | str
MultimodalContent = tuple[MultimodalContentItem, ...] | MultimodalContentItem


def merge_multimodal_content(
    *content: MultimodalContent,
) -> tuple[MultimodalContentItem, ...]:
    result: list[MultimodalContentItem] = []
    for part in content:
        match part:
            case [*parts]:
                result.extend(parts)
            case part:
                result.append(part)

    return tuple(result)
