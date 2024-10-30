from haiway import State

from draive.generation.model.default import default_generate_model
from draive.generation.model.types import ModelGenerator

__all__ = [
    "ModelGeneration",
]


class ModelGeneration(State):
    generate: ModelGenerator = default_generate_model
