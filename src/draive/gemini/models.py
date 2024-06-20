from typing import Any, Literal, NotRequired, Required, TypedDict

from draive.parameters import DataModel, Field

__all__ = [
    "GeminiGenerationConfig",
    "GeminiGenerationRequest",
    "GeminiGenerationResult",
    "GeminiMessage",
    "GeminiTextMessageContent",
    "GeminiToolConfig",
    "GeminiToolFunctionCallingConfig",
    "GeminiFunctionCallMessageContent",
    "GeminiFunctionResponseMessageContent",
    "GeminiMessageContentBlob",
    "GeminiMessageContentReference",
    "GeminiDataMessageContent",
    "GeminiDataReferenceMessageContent",
    "GeminiFunctionResponse",
    "GeminiFunctionCall",
    "GeminiRequestMessage",
]


class GeminiMessageContentBlob(DataModel):
    mime_type: str = Field(alias="mimeType")
    data: str  # base64 encoded


class GeminiMessageContentReference(DataModel):
    mime_type: str = Field(alias="mimeType")
    uri: str = Field(alias="fileUri")


class GeminiTextMessageContent(DataModel):
    text: str


class GeminiDataMessageContent(DataModel):
    data: GeminiMessageContentBlob = Field(alias="inlineData")


class GeminiDataReferenceMessageContent(DataModel):
    reference: GeminiMessageContentReference = Field(alias="fileData")


class GeminiFunctionCall(DataModel):
    name: str
    arguments: dict[str, Any] = Field(alias="args")


class GeminiFunctionCallMessageContent(DataModel):
    function_call: GeminiFunctionCall = Field(alias="functionCall")


class GeminiFunctionResponse(DataModel):
    name: str
    response: dict[str, Any]


class GeminiFunctionResponseMessageContent(DataModel):
    function_response: GeminiFunctionResponse = Field(alias="functionResponse")


GeminiMessageContent = (
    GeminiTextMessageContent
    | GeminiFunctionCallMessageContent
    | GeminiFunctionResponseMessageContent
    | GeminiDataMessageContent
    | GeminiDataReferenceMessageContent
)


class GeminiMessage(DataModel):
    role: Literal["user", "model"]
    content: list[GeminiMessageContent] = Field(alias="parts")


class GeminiRequestMessage(TypedDict, total=False):
    role: Required[Literal["user", "model"]]
    parts: Required[list[dict[str, Any]]]


class GeminiSystemMessageContent(TypedDict, total=False):
    text: Required[str]


class GeminiSystemMessage(TypedDict, total=False):
    role: Required[Literal[""]]
    parts: Required[tuple[GeminiSystemMessageContent]]


class GeminiFunctionToolSpecification(TypedDict, total=False):
    name: Required[str]
    description: Required[str]
    parameters: Required[dict[str, Any]]


class GeminiFunctionsTool(TypedDict, total=False):
    functionDeclarations: Required[list[GeminiFunctionToolSpecification]]


class GeminiToolFunctionCallingConfig(TypedDict, total=False):
    mode: Required[Literal["AUTO", "ANY", "NONE"]]


class GeminiToolConfig(TypedDict, total=False):
    functionCallingConfig: Required[GeminiToolFunctionCallingConfig]


class GeminiGenerationConfig(TypedDict, total=False):
    responseMimeType: Required[Literal["text/plain", "application/json"]]
    responseSchema: NotRequired[dict[str, Any] | None]
    temperature: NotRequired[float]
    maxOutputTokens: NotRequired[int | None]
    topP: NotRequired[float | None]
    topK: NotRequired[int | None]
    candidateCount: NotRequired[int]


class GeminiGenerationRequest(TypedDict, total=False):
    generationConfig: Required[GeminiGenerationConfig]
    systemInstruction: NotRequired[GeminiSystemMessage | None]
    contents: Required[list[GeminiRequestMessage]]
    tools: Required[list[GeminiFunctionsTool]]
    toolConfig: Required[GeminiToolConfig]


class GeminiUsage(DataModel):
    prompt_tokens: int = Field(alias="promptTokenCount")
    generated_tokens: int = Field(alias="candidatesTokenCount")


class GeminiChoice(DataModel):
    content: GeminiMessage
    finish_reason: Literal[
        "STOP",
        "MAX_TOKENS",
        "SAFETY",
        "RECITATION",
        "OTHER",
    ] = Field(alias="finishReason")


class GeminiGenerationResult(DataModel):
    choices: list[GeminiChoice] = Field(alias="candidates")
    usage: GeminiUsage = Field(alias="usageMetadata")
