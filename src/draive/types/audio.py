from draive.types.model import Model

__all__ = [
    "AudioBase64Content",
    "AudioContent",
    "AudioDataContent",
    "AudioURLContent",
]


class AudioURLContent(Model):
    audio_url: str
    audio_transcription: str | None = None


class AudioBase64Content(Model):
    audio_base64: str
    audio_transcription: str | None = None


class AudioDataContent(Model):
    audio_data: bytes
    audio_transcription: str | None = None


AudioContent = AudioURLContent | AudioBase64Content | AudioDataContent
