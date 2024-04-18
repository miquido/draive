from draive.helpers import MISSING, Missing
from draive.types.model import Model

__all__ = [
    "VideoBase64Content",
    "VideoContent",
    "VideoURLContent",
]


class VideoURLContent(Model):
    video_url: str
    video_transcription: str | Missing = MISSING


class VideoBase64Content(Model):
    video_base64: str
    video_transcription: str | Missing = MISSING


VideoContent = VideoURLContent | VideoBase64Content
