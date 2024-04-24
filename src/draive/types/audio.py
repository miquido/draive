from draive.helpers import MISSING, Missing
from draive.types.model import Model

__all__ = [
    "AudioBase64Content",
    "AudioContent",
    "AudioURLContent",
]


class AudioURLContent(Model):
    audio_url: str
    audio_transcription: str | Missing = MISSING


class AudioBase64Content(Model):
    audio_base64: str
    audio_transcription: str | Missing = MISSING


AudioContent = AudioURLContent | AudioBase64Content
