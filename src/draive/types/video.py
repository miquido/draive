from draive.parameters import DataModel

__all__ = [
    "VideoBase64Content",
    "VideoContent",
    "VideoURLContent",
]


class VideoURLContent(DataModel):
    mime_type: str | None = None
    video_url: str
    video_transcription: str | None = None
    meta: dict[str, str | float | int | bool | None] | None = None


class VideoBase64Content(DataModel):
    mime_type: str | None = None
    video_base64: str
    video_transcription: str | None = None
    meta: dict[str, str | float | int | bool | None] | None = None


VideoContent = VideoURLContent | VideoBase64Content
