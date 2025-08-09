from collections.abc import Callable

from haiway import BasicValue

__all__ = (
    "ParameterConversion",
    "ParameterVerification",
)


type ParameterConversion[Type] = Callable[[Type], BasicValue]
type ParameterVerification[Type] = Callable[[Type], None]
