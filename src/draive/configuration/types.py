from collections.abc import Mapping
from typing import Protocol

from typing_extensions import runtime_checkable

from draive.parameters import BasicValue

__all__ = ("ConfigurationLoading",)


@runtime_checkable
class ConfigurationLoading(Protocol):
    async def __call__(
        self,
        identifier: str,
    ) -> Mapping[str, BasicValue] | None: ...
