import json
from typing import Any

from draive.parameters.specification import ParameterSpecification, ParametersSpecification

__all__ = [
    "json_schema",
    "simplified_schema",
]


def json_schema(
    specification: ParametersSpecification,
    indent: int | None = None,
) -> str:
    return json.dumps(
        specification,
        indent=indent,
    )


def simplified_schema(
    specification: ParametersSpecification,
    indent: int | None = None,
) -> str:
    match specification:
        case {"properties": {**properties}}:
            return json.dumps(
                {
                    key: _simplified_schema_property(
                        specification=specification,
                    )
                    for key, specification in properties.items()
                },
                indent=indent,
            )

        case other:
            raise ValueError("Unsupported basic specification: %s", other)


def _simplified_schema_property(  # noqa: C901, PLR0912, PLR0911
    specification: ParameterSpecification,
) -> dict[str, Any] | list[Any] | str:
    match specification:
        case {"type": "null", "description": str() as description}:
            return f"null({description})" if description else "null"

        case {"type": "null"}:
            return "null"

        case {"type": "boolean", "description": str() as description}:
            return f"boolean({description})" if description else "boolean"

        case {"type": "boolean"}:
            return "boolean"

        case {"type": "integer", "description": str() as description, "enum": selection}:
            return (
                "|".join(str(element) for element in selection) + f"({description})"
                if description
                else "|".join(str(element) for element in selection)
            )

        case {"type": "integer", "enum": selection}:
            return "|".join(str(element) for element in selection)

        case {"type": "integer", "description": str() as description}:
            return f"integer({description})" if description else "integer"

        case {"type": "integer"}:
            return "integer"

        case {"type": "number", "description": str() as description, "enum": selection}:
            return (
                "|".join(str(element) for element in selection) + f"({description})"
                if description
                else "|".join(str(element) for element in selection)
            )

        case {"type": "number", "enum": selection}:
            return "|".join(str(element) for element in selection)

        case {"type": "number", "description": str() as description}:
            return f"number({description})" if description else "number"

        case {"type": "number"}:
            return "number"

        case {"type": "string", "description": str() as description, "format": format}:
            return f"{format}({description})" if description else format

        case {"type": "string", "format": format}:
            return format

        case {"type": "string", "description": str() as description, "enum": selection}:
            return (
                "|".join(f"'{element}'" for element in selection) + f"({description})"
                if description
                else "|".join(f"'{element}'" for element in selection)
            )

        case {"type": "string", "enum": selection}:
            return "|".join(f"'{element}'" for element in selection)

        case {"type": "string", "description": str() as description}:
            return f"string({description})" if description else "string"

        case {"type": "string"}:
            return "string"

        case {"oneOf": [*alternatives], "description": str() as description}:
            alternative_elements: list[str] = []
            for alternative in alternatives:
                match _simplified_schema_property(specification=alternative):
                    case str() as property:
                        alternative_elements.append(property)

                    case list() as elements:
                        alternative_elements.append(json.dumps(elements))

                    case dict() as elements:
                        alternative_elements.append(json.dumps(elements))

            return (
                "|".join(alternative_elements) + f"({description})"
                if description
                else "|".join(alternative_elements)
            )

        case {"oneOf": [*alternatives]}:
            alternative_elements: list[str] = []
            for alternative in alternatives:
                match _simplified_schema_property(specification=alternative):
                    case str() as property:
                        alternative_elements.append(property)

                    case list() as elements:
                        alternative_elements.append(json.dumps(elements))

                    case dict() as elements:
                        alternative_elements.append(json.dumps(elements))

            return "|".join(alternative_elements)

        case {"type": "array", "items": items, "description": str() as description}:
            return [_simplified_schema_property(specification=items), f"({description})"]

        case {"type": "array", "items": items}:
            return [
                _simplified_schema_property(
                    specification=items,
                ),
            ]

        case {"type": "array", "prefixItems": items, "description": str() as description}:
            return [
                _simplified_schema_property(
                    specification=item,
                )
                for item in items
            ] + [f"({description})"]

        case {"type": "array", "prefixItems": items}:
            return [
                _simplified_schema_property(
                    specification=item,
                )
                for item in items
            ]

        case {"type": "array", "description": str() as description}:
            return [f"({description})"]

        case {"type": "array"}:
            return []

        case {"type": "object", "properties": {**properties}, "description": str() as description}:
            return {
                key: _simplified_schema_property(
                    specification=specification,
                )
                for key, specification in properties.items()
            }

        case {"type": "object", "properties": {**properties}}:
            return {
                key: _simplified_schema_property(
                    specification=specification,
                )
                for key, specification in properties.items()
            }

        case other:
            raise ValueError("Unsupported basic specification element: %s", other)
