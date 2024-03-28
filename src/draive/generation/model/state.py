from draive.generation.model.generator import ModelGenerator
from draive.generation.model.lmm import lmm_generate_model
from draive.tools import Toolbox
from draive.types import State

__all__ = [
    "ModelGeneration",
]


class ModelGeneration(State):
    generate: ModelGenerator = lmm_generate_model
    tools: Toolbox | None = None
