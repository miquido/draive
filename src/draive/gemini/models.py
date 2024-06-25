from typing import Any, Literal, Required, TypedDict

from draive.parameters import DataModel, Field

__all__ = [
    "GeminiGenerationResult",
    "GeminiMessage",
    "GeminiTextMessageContent",
    "GeminiFunctionCallMessageContent",
    "GeminiFunctionResponseMessageContent",
    "GeminiMessageContentBlob",
    "GeminiMessageContentReference",
    "GeminiDataMessageContent",
    "GeminiDataReferenceMessageContent",
    "GeminiFunctionResponse",
    "GeminiFunctionCall",
]


class GeminiMessageContentBlob(DataModel):
    mime_type: str = Field(aliased="mimeType")
    data: str  # base64 encoded


class GeminiMessageContentReference(DataModel):
    mime_type: str = Field(aliased="mimeType")
    uri: str = Field(aliased="fileUri")


class GeminiTextMessageContent(DataModel):
    text: str


class GeminiDataMessageContent(DataModel):
    data: GeminiMessageContentBlob = Field(aliased="inlineData")


class GeminiDataReferenceMessageContent(DataModel):
    reference: GeminiMessageContentReference = Field(aliased="fileData")


class GeminiFunctionCall(DataModel):
    name: str
    arguments: dict[str, Any] = Field(aliased="args")


class GeminiFunctionCallMessageContent(DataModel):
    function_call: GeminiFunctionCall = Field(aliased="functionCall")


class GeminiFunctionResponse(DataModel):
    name: str
    response: dict[str, Any]


class GeminiFunctionResponseMessageContent(DataModel):
    function_response: GeminiFunctionResponse = Field(aliased="functionResponse")


GeminiMessageContent = (
    GeminiTextMessageContent
    | GeminiFunctionCallMessageContent
    | GeminiFunctionResponseMessageContent
    | GeminiDataMessageContent
    | GeminiDataReferenceMessageContent
)


class GeminiMessage(DataModel):
    role: Literal["user", "model"]
    content: list[GeminiMessageContent] = Field(aliased="parts")


class GeminiRequestMessage(TypedDict, total=False):
    role: Required[Literal["user", "model"]]
    parts: Required[list[dict[str, Any]]]


class GeminiSystemMessageContent(TypedDict, total=False):
    text: Required[str]


class GeminiSystemMessage(TypedDict, total=False):
    parts: Required[tuple[GeminiSystemMessageContent]]


class GeminiFunctionToolSpecification(TypedDict, total=False):
    name: Required[str]
    description: Required[str]
    parameters: Required[dict[str, Any]]


class GeminiFunctionsTool(TypedDict, total=False):
    functionDeclarations: Required[list[GeminiFunctionToolSpecification]]


class GeminiUsage(DataModel):
    prompt_tokens: int = Field(aliased="promptTokenCount", default_factory=int)
    generated_tokens: int = Field(aliased="candidatesTokenCount", default_factory=int)


class GeminiChoice(DataModel):
    content: GeminiMessage | None = None
    finish_reason: Literal[
        "STOP",
        "MAX_TOKENS",
        "SAFETY",
        "RECITATION",
        "OTHER",
    ] = Field(aliased="finishReason")


class GeminiGenerationResult(DataModel):
    choices: list[GeminiChoice] = Field(aliased="candidates")
    usage: GeminiUsage = Field(aliased="usageMetadata")
