import types
import typing
from json import loads
from typing import Any, cast, final, get_args, get_origin

from draive.scope import ctx
from draive.tools import Tool, ToolException
from draive.tools.state import ToolCallContext
from draive.types import Model, StreamingProgressUpdate, StringConvertible, ToolSpecification

__all__ = [
    "Toolbox",
]

AnyTool = Tool[Any, Any]


@final
class Toolbox:
    def __init__(
        self,
        *tools: AnyTool,
    ) -> None:
        self._tools: dict[str, AnyTool] = {tool.name: tool for tool in tools}

    @property
    def available_tools(self) -> list[ToolSpecification]:
        return [tool.specification for tool in self._tools.values() if tool.available]

    async def call_tool(
        self,
        name: str,
        /,
        call_id: str,
        arguments: str | bytes | None,
        progress: StreamingProgressUpdate[Model] | None = None,
    ) -> StringConvertible:
        if tool := self._tools[name]:
            with ctx.updated(
                ToolCallContext(
                    call_id=call_id,
                    progress=progress or (lambda update: None),
                )
            ):
                validated_args: dict[str, Any]
                try:
                    validated_args = validated_arguments(
                        annotations=tool._call_annotations,  # pyright: ignore[reportPrivateUsage]
                        values=loads(arguments) if arguments else {},
                    )
                except (ValueError, TypeError) as exc:
                    raise ToolException("Requested tool arguments invalid", name) from exc

                return await tool(**validated_args)
        else:
            raise ToolException("Requested tool is not defined", name)


def validated_arguments(
    annotations: dict[str, Any],
    values: dict[str, Any],
) -> dict[str, Any]:
    # TODO: allow alternative names / aliases through Annotated
    for key, value in values.items():
        if annotation := annotations.get(key):
            if validated := _validated(
                annotation=annotation,
                value=value,
            ):
                values[key] = validated
            else:
                continue  # keep it as is
        else:
            raise ValueError("Unexpected value", key, value)

    # missing and default values are handled by call
    return values


def _validated(  # noqa: PLR0911
    annotation: Any,
    value: Any,
) -> Any | None:
    # TODO: allow custom validations through Annotated
    match get_origin(annotation) or annotation:
        case typing.Annotated:
            match get_args(annotation):
                case [annotated, *_]:
                    return _validated(
                        annotation=annotated,
                        value=value,
                    )

                case annotated:
                    raise TypeError("Unsupported annotated type", annotated)
        case typing.Literal:
            if value in get_args(annotation):
                return None
            else:
                raise TypeError("Invalid value", annotation, value)
        case types.UnionType | typing.Union:
            for alternative in get_args(annotation):
                try:
                    return _validated(
                        annotation=alternative,
                        value=value,
                    )
                except TypeError:
                    continue  # check next alternative

            raise TypeError("Invalid value", annotation, value)

        case expected_type:
            if isinstance(value, expected_type):
                return None  # keep it as is
            elif isinstance(value, dict) and issubclass(expected_type, Model):
                return expected_type.from_dict(value=cast(dict[str, Any], value))
            elif isinstance(value, float) and expected_type == int:
                return int(value)  # auto convert float to int
            elif isinstance(value, int) and expected_type == float:
                return float(value)  # auto convert int to float
            else:
                raise TypeError("Invalid value", expected_type, value)  # pyright: ignore[reportUnknownArgumentType]
