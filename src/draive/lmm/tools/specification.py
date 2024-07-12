from typing import Literal, Required, TypedDict

from draive.parameters import ParametersSpecification

__all__ = [
    "ToolFunctionSpecification",
    "ToolSpecification",
]


class ToolFunctionSpecification(TypedDict, total=False):
    name: Required[str]
    description: Required[str]
    parameters: Required[ParametersSpecification]


class ToolSpecification(TypedDict, total=False):
    type: Required[Literal["function"]]
    function: Required[ToolFunctionSpecification]
