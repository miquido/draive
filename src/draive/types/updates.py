from typing import Generic, Protocol, TypeVar

from draive.types.model import Model

__all__ = [
    "UpdateSend",
]


_Update_contra = TypeVar(
    "_Update_contra",
    contravariant=True,
    bound=Model | str,
)


class UpdateSend(Protocol, Generic[_Update_contra]):
    def __call__(
        self,
        update: _Update_contra,
    ):
        ...
