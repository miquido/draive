from draive.types import MultimodalContent

__all__ = [
    "ToolException",
    "ToolError",
]


class ToolException(Exception):
    pass


class ToolError(ToolException):
    def __init__(
        self,
        *args: object,
        content: MultimodalContent,
    ) -> None:
        super().__init__(*args)
        self.content: MultimodalContent = content
