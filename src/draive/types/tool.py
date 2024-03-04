from typing import Literal, NotRequired, Required, TypedDict

from draive.helpers import ParametersSpecification

__all__ = [
    "ToolSpecification",
]


class ToolFunctionSpecification(TypedDict, total=False):
    name: Required[str]
    description: NotRequired[str]
    parameters: Required[ParametersSpecification]


class ToolSpecification(TypedDict, total=False):
    type: Required[Literal["function"]]
    function: Required[ToolFunctionSpecification]
