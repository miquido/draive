from typing import Any, Literal

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
    role: Literal["user", "model", ""]
    content: list[GeminiMessageContent] = Field(alias="parts")


class GeminiFunctionToolSpecification(DataModel):
    name: str
    description: str
    parameters: dict[str, Any]


class GeminiFunctionsTool(DataModel):
    functions: list[GeminiFunctionToolSpecification] = Field(alias="functionDeclarations")


class GeminiToolFunctionCallingConfig(DataModel):
    mode: Literal["AUTO", "ANY", "NONE"]


class GeminiToolConfig(DataModel):
    function_calling: GeminiToolFunctionCallingConfig = Field(alias="functionCallingConfig")


class GeminiGenerationConfig(DataModel):
    response_format: Literal["text/plain", "application/json"] = Field(
        alias="responseMimeType",
        default="text/plain",
    )
    response_schema: dict[str, Any] | None = Field(
        alias="responseSchema",
        default=None,
    )
    temperature: float
    max_tokens: int | None = Field(alias="maxOutputTokens", default=None)
    top_p: float | None = Field(alias="topP", default=None)
    top_k: int | None = Field(alias="topK", default=None)
    n: int = Field(alias="candidateCount", default=1)


class GeminiGenerationRequest(DataModel):
    config: GeminiGenerationConfig = Field(alias="generationConfig")
    instruction: GeminiMessage | None = Field(alias="systemInstruction", default=None)
    messages: list[GeminiMessage] = Field(alias="contents")
    tools: list[GeminiFunctionsTool] | None = None
    tools_config: GeminiToolConfig | None = Field(alias="toolConfig", default=None)


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
