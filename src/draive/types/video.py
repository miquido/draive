from draive.types.model import Model

__all__ = [
    "VideoBase64Content",
    "VideoContent",
    "VideoURLContent",
]


class VideoURLContent(Model):
    video_url: str
    video_transcription: str | None = None


class VideoBase64Content(Model):
    video_base64: str
    video_transcription: str | None = None


VideoContent = VideoURLContent | VideoBase64Content
