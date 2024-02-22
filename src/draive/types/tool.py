from typing import Literal, NotRequired, Required, TypedDict

from draive.helpers import ParametersSpecification

__all__ = [
    "ToolSpecification",
]


class ToolFunctionParametersSpecification(TypedDict, total=False):
    type: Required[Literal["object"]]
    properties: Required[ParametersSpecification]


class ToolFunctionSpecification(TypedDict, total=False):
    name: Required[str]
    description: NotRequired[str]
    parameters: Required[ToolFunctionParametersSpecification]


class ToolSpecification(TypedDict, total=False):
    type: Required[Literal["function"]]
    function: Required[ToolFunctionSpecification]
