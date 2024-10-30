from haiway import State

from draive.generation.text.default import default_generate_text
from draive.generation.text.types import TextGenerator

__all__ = [
    "TextGeneration",
]


class TextGeneration(State):
    generate: TextGenerator = default_generate_text
