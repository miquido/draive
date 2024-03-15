from draive.openai import openai_generate_model, openai_generate_text
from draive.scope import ScopeState
from draive.types import ModelGenerator, TextGenerator, Toolset

__all__ = [
    "TextGeneration",
    "ModelGeneration",
]


class TextGeneration(ScopeState):
    generate: TextGenerator = openai_generate_text
    toolset: Toolset | None = None


class ModelGeneration(ScopeState):
    generate: ModelGenerator = openai_generate_model
    toolset: Toolset | None = None
