from typing import Literal, Required, TypedDict, final

from draive.types.parameters import ParametersSpecification

__all__ = [
    "ToolSpecification",
]


@final
class ToolFunctionSpecification(TypedDict, total=False):
    name: Required[str]
    description: Required[str]
    parameters: Required[ParametersSpecification]


@final
class ToolSpecification(TypedDict, total=False):
    type: Required[Literal["function"]]
    function: Required[ToolFunctionSpecification]
