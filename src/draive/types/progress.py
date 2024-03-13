from typing import Generic, Protocol, TypeVar

from draive.types.model import Model

__all__ = [
    "ProgressUpdate",
]


_Progress_contra = TypeVar(
    "_Progress_contra",
    contravariant=True,
    bound=Model,
)


class ProgressUpdate(Protocol, Generic[_Progress_contra]):
    def __call__(
        self,
        update: _Progress_contra,
    ):
        ...
