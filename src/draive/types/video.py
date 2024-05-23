from draive.types.model import Model

__all__ = [
    "VideoBase64Content",
    "VideoContent",
    "VideoDataContent",
    "VideoURLContent",
]


class VideoURLContent(Model):
    video_url: str
    video_transcription: str | None = None


class VideoBase64Content(Model):
    video_base64: str
    video_transcription: str | None = None


class VideoDataContent(Model):
    video_data: bytes
    video_transcription: str | None = None


VideoContent = VideoURLContent | VideoBase64Content | VideoDataContent
