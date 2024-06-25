from typing import Literal

from draive import DataModel, Field


class SchemaNestedModel(DataModel):
    value: Literal["A", "B", "C"] = Field(description="selection")
    other_value: None


class SchemaModel(DataModel):
    str_value: str
    int_value: int
    float_value: float
    bool_value: bool
    list_value: list[str]
    optional_value: str | None = Field(description="alternative")
    nested: SchemaNestedModel = Field(aliased="nested_value", description="alternative")


json_schema: str = """\
{
  "type": "object",
  "properties": {
    "str_value": {
      "type": "string"
    },
    "int_value": {
      "type": "integer"
    },
    "float_value": {
      "type": "number"
    },
    "bool_value": {
      "type": "boolean"
    },
    "list_value": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "optional_value": {
      "oneOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "description": "alternative"
    },
    "nested": {
      "type": "object",
      "properties": {
        "value": {
          "type": "string",
          "enum": [
            "A",
            "B",
            "C"
          ],
          "description": "selection"
        },
        "other_value": {
          "type": "null"
        }
      },
      "required": [
        "value",
        "other_value"
      ],
      "description": "alternative"
    }
  },
  "required": [
    "str_value",
    "int_value",
    "float_value",
    "bool_value",
    "list_value",
    "optional_value",
    "nested"
  ]
}\
"""


def test_json_schema() -> None:
    assert json_schema == SchemaModel.json_schema(indent=2)


simplified_schema: str = """\
{
  "str_value": "string",
  "int_value": "integer",
  "float_value": "number",
  "bool_value": "boolean",
  "list_value": [
    "string"
  ],
  "optional_value": "string|null(alternative)",
  "nested": {
    "value": "'A'|'B'|'C'(selection)",
    "other_value": "null"
  }
}\
"""


def test_simplified_schema() -> None:
    assert simplified_schema == SchemaModel.simplified_schema(indent=2)
