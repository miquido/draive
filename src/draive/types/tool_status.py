# from typing import Literal

# from draive.parameters.model import DataModel
# from draive.types.multimodal import MultimodalContent

# __all__ = [
#     "ToolCallStatus",
# ]


# class ToolCallStatus(DataModel):
#     identifier: str
#     tool: str
#     status: Literal[
#         "STARTED",
#         "RUNNING",
#         "FINISHED",
#         "FAILED",
#     ]
#     content: MultimodalContent | None = None
