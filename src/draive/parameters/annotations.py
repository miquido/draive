import types
import typing
from collections.abc import Callable, Iterator
from types import GenericAlias, NoneType, UnionType
from typing import (
    Any,
    ForwardRef,
    ParamSpec,
    TypeAliasType,
    TypeVar,
    TypeVarTuple,
    _UnionGenericAlias,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
    _UnpackGenericAlias,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
    get_args,
    get_origin,
    get_type_hints,
)

from haiway.types import missing

__all__ = [
    "object_annotations",
    "resolve_annotation",
    "allows_missing",
    "ParameterDefaultFactory",
]


type ParameterDefaultFactory[Value] = Callable[[], Value]


def object_annotations(
    annotation: Any,
    /,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
) -> dict[str, Any]:
    return get_type_hints(
        annotation,
        globalns=globalns,
        localns=localns,
        include_extras=False,
    )


def allows_missing(
    annotation: Any,
    /,
    type_arguments: dict[str, Any],
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
) -> bool:
    resolved_type, resolved_args = resolve_annotation(
        annotation,
        type_arguments=type_arguments,
        globalns=globalns,
        localns=localns,
    )

    match resolved_type:
        case missing.Missing:
            return True

        case types.UnionType:
            return missing.Missing in resolved_args

        case _:
            return False


def resolve_annotation(  # noqa: PLR0911, C901, PLR0912
    annotation: Any,
    /,
    type_arguments: dict[str, Any],
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
) -> tuple[Any, tuple[Any, ...]]:
    match annotation:
        case NoneType() | None:
            return (
                NoneType,
                (),
            )

        case type() as type_annotation:
            return (
                get_origin(type_annotation) or type_annotation,
                get_args(type_annotation),
            )

        case UnionType() as union_annotation:
            return (
                UnionType,  # pyright: ignore[reportReturnType]
                get_args(union_annotation),
            )

        case _UnionGenericAlias() as union_alias_annotation:  # pyright: ignore[reportUnknownVariableType]
            return (
                UnionType,  # pyright: ignore[reportReturnType]
                get_args(union_alias_annotation),
            )

        case TypeAliasType() as alias_annotation:
            return resolve_annotation(
                alias_annotation.__value__,
                type_arguments=type_arguments,
                globalns=globalns,
                localns=localns,
            )

        case GenericAlias() as generic_annotation:
            match generic_annotation.__origin__:
                case TypeAliasType() as alias:  # pyright: ignore[reportUnnecessaryComparison]
                    resolved_args: Iterator[Any] = generic_annotation.__args__.__iter__()
                    # TODO: only basic cases are covered currently
                    merged_arguments: list[Any] = []
                    alias_value: Any = alias.__value__
                    for arg in get_args(alias_value):
                        match arg:
                            case TypeVar():
                                merged_arguments.append(next(resolved_args))

                            case _UnpackGenericAlias():
                                merged_arguments.extend(resolved_args)

                            case other:
                                merged_arguments.append(other)

                    return (
                        get_origin(alias_value) or alias_value,
                        tuple(merged_arguments),
                    )

                case parametrized if hasattr(parametrized, "__PARAMETERS__"):
                    resolved_parameters: tuple[Any, ...] = tuple(
                        type_arguments.get(
                            arg.__name__,
                            arg.__bound__ or Any,
                        )
                        if isinstance(arg, TypeVar)
                        else arg
                        for arg in generic_annotation.__args__
                    )
                    # ParametrizedData subtypes always have __class_getitem__
                    resolved_type: type[Any] = parametrized[resolved_parameters]  # pyright: ignore[reportIndexIssue, reportUnknownVariableType]

                    return (
                        resolved_type,  # pyright: ignore[reportUnknownVariableType]
                        get_args(resolved_type),
                    )

                case origin:
                    return (
                        origin,
                        generic_annotation.__args__,
                    )

        case TypeVar() as variable_annotation:
            parameter_type: Any = type_arguments.get(
                variable_annotation.__name__,
                variable_annotation.__bound__ or Any,
            )

            match parameter_type:
                case typing.TypeVar():
                    return resolve_annotation(
                        variable_annotation.__bound__ or Any,
                        type_arguments=type_arguments,
                        globalns=globalns,
                        localns=localns,
                    )

                case other:
                    return resolve_annotation(
                        other,
                        type_arguments=type_arguments,
                        globalns=globalns,
                        localns=localns,
                    )

        case ParamSpec():
            raise NotImplementedError("ParamSpec type parameter is not supported yet")

        case TypeVarTuple():
            raise NotImplementedError("TypeVarTuple type parameter is not supported yet")

        case str() as str_annotation:
            return resolve_annotation(
                ForwardRef(str_annotation)._evaluate(  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                    globalns=globalns,
                    localns=localns,
                    recursive_guard=frozenset(),
                ),
                type_arguments=type_arguments,
                globalns=globalns,
                localns=localns,
            )

        case ForwardRef() as ref_annotation:
            return resolve_annotation(
                ref_annotation._evaluate(  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                    globalns=globalns,
                    localns=localns,
                    recursive_guard=frozenset(),
                ),
                type_arguments=type_arguments,
                globalns=globalns,
                localns=localns,
            )

        case other:
            match get_origin(other):
                case typing.Annotated | typing.Final | typing.Required | typing.NotRequired:
                    return resolve_annotation(
                        get_args(other)[0],
                        type_arguments=type_arguments,
                        globalns=globalns,
                        localns=localns,
                    )

                case None:
                    return (
                        other,
                        get_args(other),
                    )

                case other_origin:
                    return (
                        other_origin,
                        get_args(other),
                    )
