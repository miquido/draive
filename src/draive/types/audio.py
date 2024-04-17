from draive.types.model import Model

__all__ = [
    "AudioBase64Content",
    "AudioContent",
    "AudioURLContent",
]


class AudioURLContent(Model):
    audio_url: str


class AudioBase64Content(Model):
    audio_base64: str


AudioContent = AudioURLContent | AudioBase64Content
