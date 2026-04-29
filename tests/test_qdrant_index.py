import pytest

from draive.embedding import VectorIndex
from draive.qdrant.index import QdrantVectorIndex


def test_qdrant_vector_index_call_remains_compatible() -> None:
    with pytest.raises(
        RuntimeError,
        match="QdrantVectorIndex instantiation is forbidden",
    ):
        QdrantVectorIndex()


def test_qdrant_vector_index_prepare_returns_vector_index() -> None:
    assert isinstance(QdrantVectorIndex.prepare(), VectorIndex)
