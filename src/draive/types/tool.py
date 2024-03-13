from typing import Literal, Required, TypedDict, final

from draive.types.model import Model
from draive.types.parameters import ParametersSpecification

__all__ = [
    "ToolCallStatus",
    "ToolCallProgress",
    "ToolSpecification",
    "ToolException",
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


ToolCallStatus = Literal["STARTED", "RUNNING", "FINISHED", "FAILED"]


class ToolCallProgress(Model):
    call_id: str
    tool: str
    status: ToolCallStatus
    content: Model | None


class ToolException(Exception):
    pass
