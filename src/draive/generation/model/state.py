from draive.generation.model.generator import ModelGenerator
from draive.generation.model.lmm import lmm_generate_model
from draive.parameters import State

__all__ = [
    "ModelGeneration",
]


class ModelGeneration(State):
    generate: ModelGenerator = lmm_generate_model
