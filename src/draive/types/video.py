from draive.types.model import Model

__all__ = [
    "VideoBase64Content",
    "VideoContent",
    "VideoURLContent",
]


class VideoURLContent(Model):
    video_url: str


class VideoBase64Content(Model):
    video_base64: str


VideoContent = VideoURLContent | VideoBase64Content
