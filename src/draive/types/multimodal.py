from draive.types.images import ImageContent

__all__ = [
    "MultimodalContent",
]

MultimodalContentItem = ImageContent | str
MultimodalContent = list[MultimodalContentItem] | MultimodalContentItem
