from draive.parameters.model import DataModel

__all__ = [
    "AudioBase64Content",
    "AudioContent",
    "AudioDataContent",
    "AudioURLContent",
]


class AudioURLContent(DataModel):
    mime_type: str | None = None
    audio_url: str
    audio_transcription: str | None = None


class AudioBase64Content(DataModel):
    mime_type: str | None = None
    audio_base64: str
    audio_transcription: str | None = None


class AudioDataContent(DataModel):
    mime_type: str | None = None
    audio_data: bytes
    audio_transcription: str | None = None


AudioContent = AudioURLContent | AudioBase64Content | AudioDataContent
