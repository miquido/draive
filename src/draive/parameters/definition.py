import json
import sys
from collections.abc import Callable, Generator, Iterable, Iterator
from functools import cached_property
from typing import Any, final

from draive.parameters.specification import (
    ParameterSpecification,
    ParametersSpecification,
    parameter_specification,
)
from draive.parameters.validation import (
    ParameterValidationContext,
    ParameterValidationError,
    ParameterValidator,
)
from draive.utils import MISSING, Missing, freeze, missing, not_missing

__all__ = [
    "ParameterDefinition",
    "ParametersDefinition",
]


@final
class ParameterDefinition[Value]:
    def __init__(  # noqa: PLR0913
        self,
        name: str,
        alias: str | None,
        description: str | None,
        annotation: Any,
        default: Value | Missing,
        default_factory: Callable[[], Value] | Missing,
        validator: ParameterValidator[Value],
        specification: ParameterSpecification | Missing,
    ) -> None:
        assert name != alias, "Alias can't be the same as name"  # nosec: B101
        assert (  # nosec: B101
            missing(default_factory) or missing(default)
        ), "Can't specify both default value and factory"

        self.name: str = name
        self.alias: str | None = alias
        self.description: str | None = description
        self.annotation: Any = annotation
        self.default_value: Callable[[], Value | Missing]
        if not_missing(default_factory):
            self.default_value = default_factory

        elif not_missing(default):
            self.default_value = lambda: default

        else:
            self.default_value = lambda: MISSING

        self.has_default: bool = not_missing(default_factory) or not_missing(default)
        self.validator: ParameterValidator[Value] = validator
        self.specification: ParameterSpecification | Missing = specification

    def validated_value(
        self,
        value: Any,
        /,
        context: ParameterValidationContext,
    ) -> Any:
        if missing(value):
            return self.validator(self.default_value(), context)

        else:
            return self.validator(value, context)


@final
class ParametersDefinition:
    def __init__(
        self,
        source: type[Any] | str,
        /,
        parameters: Generator[ParameterDefinition[Any], None, None]
        | Iterator[ParameterDefinition[Any]]
        | Iterable[ParameterDefinition[Any]],
    ) -> None:
        self.source: type[Any] | str = source
        self.parameters: list[ParameterDefinition[Any]] = []
        self._alias_map: dict[str, str] = {}
        names: set[str] = set()
        aliases: set[str] = set()
        for parameter in parameters:
            assert (  # nosec: B101
                parameter.name not in names
            ), f"Parameter names can't overlap with each other: {parameter.name}"
            assert (  # nosec: B101
                parameter.name not in aliases
            ), f"Parameter aliases can't overlap with names: {parameter.name}"
            names.add(parameter.name)
            if alias := parameter.alias:
                assert alias not in names, f"Parameter aliases can't overlap with names: {alias}"  # nosec: B101
                assert alias not in aliases, f"Duplicate parameter aliases are not allowed: {alias}"  # nosec: B101
                aliases.add(alias)
                self._alias_map[parameter.name] = alias

            self.parameters.append(parameter)

        assert names.isdisjoint(aliases), "Aliases can't overlap regular names"  # nosec: B101

        freeze(self)

    def validated(
        self,
        context: ParameterValidationContext,
        **parameters: Any,
    ) -> dict[str, Any]:
        validated: dict[str, Any] = {}
        for parameter in self.parameters:
            parameter_context: ParameterValidationContext = (*context, f".{parameter.name}")
            if parameter.name in parameters:
                validated[parameter.name] = parameter.validated_value(
                    parameters.get(parameter.name, MISSING),
                    context=parameter_context,
                )

            elif (alias := parameter.alias) and (alias in parameters):
                validated[parameter.name] = parameter.validated_value(
                    parameters.get(alias, MISSING),
                    context=parameter_context,
                )

            else:
                try:
                    validated[parameter.name] = parameter.validated_value(
                        MISSING,
                        context=parameter_context,
                    )

                except ParameterValidationError:
                    raise ParameterValidationError.missing(context=parameter_context) from None

        return validated

    def aliased(
        self,
        **parameters: Any,
    ) -> dict[str, Any]:
        return {self._alias_map.get(key, key): value for key, value in parameters.items()}

    @cached_property
    def specification(self) -> ParametersSpecification | Missing:
        globalns: dict[str, Any]
        localns: dict[str, Any] | None
        recursion_guard: frozenset[Any]
        if isinstance(self.source, str):
            globalns = sys.modules.get(self.source).__dict__
            localns = None
            recursion_guard = frozenset()
        else:
            globalns = sys.modules.get(self.source.__module__).__dict__
            localns = {self.source.__name__: self.source}
            recursion_guard = frozenset({self.source})

        aliased_required: list[str] = []
        properties: dict[str, ParameterSpecification] = {}
        for parameter in self.parameters:
            if not_missing(parameter.specification):
                properties[parameter.alias or parameter.name] = parameter.specification

            else:
                resolved_specification: ParameterSpecification | Missing = parameter_specification(
                    annotation=parameter.annotation,
                    description=parameter.description,
                    globalns=globalns,
                    localns=localns,
                    recursion_guard=recursion_guard,
                )

                if not_missing(resolved_specification):
                    parameter.specification = resolved_specification
                    properties[parameter.alias or parameter.name] = resolved_specification

                else:
                    # when at least one element can't be represented in specification
                    # then the whole thing can't be represented in specification
                    return MISSING

            if not parameter.has_default:
                aliased_required.append(parameter.alias or parameter.name)

        return {
            "type": "object",
            "properties": properties,
            "required": aliased_required,
        }

    @cached_property
    def json_schema(self) -> str:
        if not_missing(self.specification):
            return json.dumps(
                self.specification,
                indent=2,
            )

        else:
            raise TypeError(f"{self.__class__.__qualname__} can't be represented using JSON schema")
