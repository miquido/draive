from draive import DataModel

__all__ = [
    "FastembedTextConfig",
    "FastembedImageConfig",
]


class FastembedTextConfig(DataModel):
    model: str = "nomic-ai/nomic-embed-text-v1.5"
    cache_dir: str | None = "./embedding_models/"


class FastembedImageConfig(DataModel):
    model: str = "Qdrant/resnet50-onnx"
    cache_dir: str | None = "./embedding_models/"
