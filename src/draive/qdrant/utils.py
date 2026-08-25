from collections.abc import Callable, Generator, Mapping, Sequence, Set
from contextlib import contextmanager
from inspect import Parameter, signature
from typing import Any

from draive.qdrant.types import QdrantException

__all__ = (
    "qdrant_arguments",
    "qdrant_operation",
    "qdrant_vector",
)


def qdrant_arguments(
    target: Callable[..., Any],
    /,
    **extra: Any,
) -> Mapping[str, Any]:
    """Verify additional arguments before forwarding them to the Qdrant client.

    Client methods declare ``**kwargs`` only to assert on it, which surfaces
    unrecognized arguments as an ``AssertionError`` - or silently drops them
    when running with assertions disabled.
    """
    if not extra:
        return extra

    supported: Set[str] = {
        name
        for name, parameter in signature(target).parameters.items()
        if parameter.kind
        in (
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        )
    }
    unsupported: Sequence[str] = tuple(sorted(key for key in extra if key not in supported))
    if unsupported:
        raise ValueError(
            f"Unsupported Qdrant {target.__name__} arguments: {', '.join(unsupported)}"
        )

    return extra


@contextmanager
def qdrant_operation(
    operation: str,
    /,
    collection: str | None = None,
) -> Generator[None]:
    """Translate Qdrant client errors into ``QdrantException``.

    The client communicates over gRPC by default, which surfaces failures as
    ``grpc`` errors carrying no indication of the failing operation.

    Parameters
    ----------
    operation:
        Name of the executed operation, included within the error message.
    collection:
        Name of the collection the operation was executed on, when applicable.

    Raises
    ------
    QdrantException
        When the wrapped operation raises any error.
    """
    try:
        yield

    except QdrantException:
        raise  # already translated

    except Exception as exc:
        raise QdrantException(
            f"Qdrant {operation} failed"
            if collection is None
            else f"Qdrant {operation} failed for {collection} collection"
        ) from exc


def qdrant_vector(
    vector: Any,
    /,
) -> Sequence[float]:
    """Verify a dense vector received from Qdrant.

    Raises
    ------
    QdrantException
        When the vector is missing or is not a dense vector.
    """
    match vector:
        case [*elements]:
            if not all(isinstance(element, float) for element in elements):
                raise QdrantException("Invalid Qdrant vector element")

            return tuple(elements)

        case None:
            raise QdrantException("Missing Qdrant vector")

        case _:
            # named or sparse vectors are not supported - a single dense vector is
            raise QdrantException("Unsupported Qdrant vector")
