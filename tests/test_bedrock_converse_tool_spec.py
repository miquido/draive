from typing import Any

import pytest
from botocore.exceptions import ParamValidationError  # pyright: ignore[reportMissingModuleSource]
from botocore.session import get_session  # pyright: ignore[reportMissingModuleSource]
from botocore.validate import validate_parameters  # pyright: ignore[reportMissingModuleSource]

from draive.bedrock.converse import (  # pyright: ignore[reportPrivateUsage]
    _context_messages,
    _tools_as_tool_config,
)
from draive.models import ModelInput, ModelOutput, ModelToolRequest, ModelToolSpecification
from draive.multimodal import MultimodalContent


def _validate_converse_parameters(
    tool_config: dict[str, Any] | None,
) -> None:
    # botocore validates the request parameters locally, before any network access
    operation = get_session().get_service_model("bedrock-runtime").operation_model("Converse")
    parameters: dict[str, Any] = {
        "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "messages": [{"role": "user", "content": [{"text": "hello"}]}],
    }
    if tool_config:
        parameters["toolConfig"] = tool_config

    validate_parameters(parameters, operation.input_shape)


def test_empty_tool_description_is_rejected_by_bedrock() -> None:
    # guards the constraint the conversion has to respect
    with pytest.raises(ParamValidationError):
        _validate_converse_parameters(
            {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "probe",
                            "description": "",
                            "inputSchema": {"json": {}},
                        }
                    }
                ],
                "toolChoice": {"auto": {}},
            }
        )


def test_tool_config_without_description_passes_validation() -> None:
    tool_config = _tools_as_tool_config(
        [ModelToolSpecification.of(name="probe")],
        tool_selection="auto",
    )

    assert tool_config is not None
    tool_spec = tool_config["tools"][0]["toolSpec"]
    assert "description" not in tool_spec
    # missing parameters have to be replaced with an empty object schema
    assert tool_spec["inputSchema"] == {
        "json": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    }

    _validate_converse_parameters(tool_config)


def test_tool_config_preserves_available_description() -> None:
    tool_config = _tools_as_tool_config(
        [
            ModelToolSpecification.of(
                name="probe",
                description="Verifies things",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            )
        ],
        tool_selection="required",
    )

    assert tool_config is not None
    tool_spec = tool_config["tools"][0]["toolSpec"]
    assert tool_spec["description"] == "Verifies things"
    assert tool_spec["inputSchema"]["json"]["properties"] == {"value": {"type": "string"}}
    # sequences within the schema have to be plain lists, tuples fail the validation
    assert tool_spec["inputSchema"]["json"]["required"] == ["value"]

    _validate_converse_parameters(tool_config)


def test_tool_request_arguments_pass_validation() -> None:
    messages = _context_messages(
        (
            ModelInput.of(MultimodalContent.of("check")),
            ModelOutput.of(
                ModelToolRequest.of(
                    "call-1",
                    tool="probe",
                    arguments={"values": ["a", "b"], "nested": {"flags": [True, False]}},
                )
            ),
        )
    )

    tool_use = messages[-1]["content"][0]["toolUse"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert tool_use["input"] == {
        "values": ["a", "b"],
        "nested": {"flags": [True, False]},
    }

    operation = get_session().get_service_model("bedrock-runtime").operation_model("Converse")
    validate_parameters(
        {
            "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "messages": messages,
        },
        operation.input_shape,
    )
