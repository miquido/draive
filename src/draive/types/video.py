from draive.parameters.model import DataModel

__all__ = [
    "VideoBase64Content",
    "VideoContent",
    "VideoDataContent",
    "VideoURLContent",
]


class VideoURLContent(DataModel):
    video_url: str
    video_transcription: str | None = None


class VideoBase64Content(DataModel):
    video_base64: str
    video_transcription: str | None = None


class VideoDataContent(DataModel):
    video_data: bytes
    video_transcription: str | None = None


VideoContent = VideoURLContent | VideoBase64Content | VideoDataContent
