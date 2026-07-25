from pathlib import Path

from videorag.ingestion.transcription.base import TranscriptionBackend, TranscriptSegment


class ApiTranscriptionBackend(TranscriptionBackend):
    """Placeholder for a hosted/API-based transcription backend.

    Exists to prove the TranscriptionBackend interface is genuinely swappable via
    config (TRANSCRIPTION_BACKEND=api), not just in theory. Not implemented yet —
    wiring in a real hosted Whisper API is listed as future work rather than faked
    here, since it would need a paid key this project doesn't require by default.
    """

    def __init__(self, **_kwargs) -> None:
        pass

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        raise NotImplementedError(
            "API-based transcription backend is not implemented. "
            "Set TRANSCRIPTION_BACKEND=faster_whisper to use the local backend."
        )
