from draive.generation.text.generator import TextGenerator
from draive.generation.text.lmm import lmm_generate_text
from draive.tools import Toolbox
from draive.types import State

__all__ = [
    "TextGeneration",
]


class TextGeneration(State):
    generate: TextGenerator = lmm_generate_text
    tools: Toolbox | None = None
