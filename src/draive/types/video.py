from draive.parameters.model import DataModel

__all__ = [
    "VideoBase64Content",
    "VideoContent",
    "VideoDataContent",
    "VideoURLContent",
]


class VideoURLContent(DataModel):
    mime_type: str | None = None
    video_url: str
    video_transcription: str | None = None


class VideoBase64Content(DataModel):
    mime_type: str | None = None
    video_base64: str
    video_transcription: str | None = None


class VideoDataContent(DataModel):
    mime_type: str | None = None
    video_data: bytes
    video_transcription: str | None = None


VideoContent = VideoURLContent | VideoBase64Content | VideoDataContent
