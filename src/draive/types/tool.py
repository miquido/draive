from typing import Literal, NotRequired, Required, TypedDict, final

from draive.helpers import ParametersSpecification

__all__ = [
    "ToolSpecification",
]


@final
class ToolFunctionSpecification(TypedDict, total=False):
    name: Required[str]
    description: NotRequired[str]
    parameters: Required[ParametersSpecification]


@final
class ToolSpecification(TypedDict, total=False):
    type: Required[Literal["function"]]
    function: Required[ToolFunctionSpecification]
