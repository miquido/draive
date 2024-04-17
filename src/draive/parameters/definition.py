import json
import sys
from collections.abc import Callable, Generator, Iterable, Iterator
from functools import cached_property
from typing import Any, final

from draive.helpers import freeze
from draive.parameters.missing import MISSING_PARAMETER, MissingParameter
from draive.parameters.specification import (
    ParameterSpecification,
    ParametersSpecification,
    parameter_specification,
)

__all__ = [
    "ParameterDefinition",
    "ParametersDefinition",
]


@final  # TODO: allow draive.helpers.Missing populated with MISSING regardless of default
class ParameterDefinition[Value]:
    def __init__(  # noqa: PLR0913
        self,
        name: str,
        alias: str | None,
        description: str | None,
        annotation: Any,
        default: Value | MissingParameter,
        default_factory: Callable[[], Value] | None,
        validator: Callable[[Any], Value],
        specification: ParameterSpecification | None,
    ) -> None:
        assert name != alias, "Alias can't be the same as name"  # nosec: B101
        assert (  # nosec: B101
            default_factory is None or default is MISSING_PARAMETER
        ), "Can't specify both default value and factory"

        self.name: str = name
        self.alias: str | None = alias
        self.description: str | None = description
        self.annotation: Any = annotation
        self.default: Value | MissingParameter = default
        self.default_factory: Callable[[], Value] | None = default_factory
        self.has_default: bool = default_factory is not None or default is not MISSING_PARAMETER
        self.validator: Callable[[Any], Value] = validator
        self.specification: ParameterSpecification | None = specification

    def default_value(self) -> Any | MissingParameter:
        if factory := self.default_factory:
            return factory()
        else:
            return self.default

    def validated_value(
        self,
        value: Any | MissingParameter,
        /,
    ) -> Any:
        if value is MISSING_PARAMETER:
            default: Any | MissingParameter = self.default_value()
            if default is MISSING_PARAMETER:
                raise ValueError("Missing required parameter: `%s`", self.name)

            else:
                return self.validator(default)

        else:
            return self.validator(value)


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
        self.aliased_required: list[str] = []
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
            if not parameter.has_default:
                self.aliased_required.append(parameter.alias or parameter.name)

        assert names.isdisjoint(aliases), "Aliases can't overlap regular names"  # nosec: B101

        freeze(self)

    def validated(
        self,
        **parameters: Any,
    ) -> dict[str, Any]:
        validated: dict[str, Any] = {}
        for parameter in self.parameters:
            if parameter.name in parameters:
                validated[parameter.name] = parameter.validated_value(
                    parameters.get(parameter.name)
                )
            elif (alias := parameter.alias) and (alias in parameters):
                validated[parameter.name] = parameter.validated_value(parameters.get(alias))
            else:
                validated[parameter.name] = parameter.validated_value(MISSING_PARAMETER)

        return validated

    def aliased(
        self,
        **parameters: Any,
    ) -> dict[str, Any]:
        return {self._alias_map.get(key, key): value for key, value in parameters.items()}

    @cached_property
    def parameters_specification(self) -> ParametersSpecification:
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

        properties: dict[str, ParameterSpecification] = {}
        for parameter in self.parameters:
            if specification := parameter.specification:
                properties[parameter.alias or parameter.name] = specification
            else:
                parameter.specification = parameter_specification(
                    annotation=parameter.annotation,
                    description=parameter.description,
                    globalns=globalns,
                    localns=localns,
                    recursion_guard=recursion_guard,
                )
                properties[parameter.alias or parameter.name] = parameter.specification

        return {
            "type": "object",
            "properties": properties,
            "required": self.aliased_required,
        }

    @cached_property
    def specification(self) -> str:
        return json.dumps(
            self.parameters_specification,
            indent=2,
        )
