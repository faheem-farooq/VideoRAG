from videorag.ingestion.transcription.base import TranscriptionBackend, TranscriptSegment
from videorag.ingestion.transcription.faster_whisper import FasterWhisperBackend


def get_transcription_backend(backend_name: str, **kwargs) -> TranscriptionBackend:
    if backend_name == "faster_whisper":
        return FasterWhisperBackend(**kwargs)
    if backend_name == "api":
        from videorag.ingestion.transcription.api_backend import ApiTranscriptionBackend

        return ApiTranscriptionBackend(**kwargs)
    raise ValueError(f"Unknown transcription backend: {backend_name!r}")


__all__ = [
    "TranscriptionBackend",
    "TranscriptSegment",
    "FasterWhisperBackend",
    "get_transcription_backend",
]
