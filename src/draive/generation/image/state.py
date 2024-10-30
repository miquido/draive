from haiway import State

from draive.generation.image.typing import ImageGenerator

__all__ = [
    "ImageGeneration",
]


class ImageGeneration(State):
    generate: ImageGenerator
