from draive.types.audio import AudioContent
from draive.types.images import ImageContent
from draive.types.video import VideoContent

__all__ = [
    "MultimodalContent",
]

MultimodalContentItem = VideoContent | ImageContent | AudioContent | str
MultimodalContent = list[MultimodalContentItem] | MultimodalContentItem
