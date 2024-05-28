from draive.parameters.model import DataModel

__all__ = [
    "AudioBase64Content",
    "AudioContent",
    "AudioDataContent",
    "AudioURLContent",
]


class AudioURLContent(DataModel):
    audio_url: str
    audio_transcription: str | None = None


class AudioBase64Content(DataModel):
    audio_base64: str
    audio_transcription: str | None = None


class AudioDataContent(DataModel):
    audio_data: bytes
    audio_transcription: str | None = None


AudioContent = AudioURLContent | AudioBase64Content | AudioDataContent
