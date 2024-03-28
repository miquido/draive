from draive.types.images import ImageContent
from draive.types.model import Model

__all__ = [
    "MultimodalContent",
]

MultimodalContentItem = ImageContent | Model | str
MultimodalContent = list[MultimodalContentItem] | MultimodalContentItem
