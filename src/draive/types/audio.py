from draive.parameters import DataModel

__all__ = [
    "AudioBase64Content",
    "AudioContent",
    "AudioURLContent",
]


class AudioURLContent(DataModel):
    mime_type: str | None = None
    audio_url: str
    audio_transcription: str | None = None
    meta: dict[str, str | float | int | bool | None] | None = None


class AudioBase64Content(DataModel):
    mime_type: str | None = None
    audio_base64: str
    audio_transcription: str | None = None
    meta: dict[str, str | float | int | bool | None] | None = None


AudioContent = AudioURLContent | AudioBase64Content
