from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranscriptSegment:
    """A single timestamped unit of speech as produced by a transcription backend."""

    start: float
    end: float
    text: str


class TranscriptionBackend(ABC):
    """Interface all transcription backends must implement.

    Kept deliberately narrow so the default local faster-whisper backend and any
    future API-based backend (e.g. a hosted Whisper API) are interchangeable via
    config, with no changes needed elsewhere in the ingestion pipeline.
    """

    @abstractmethod
    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        """Return timestamped segments in chronological order."""
        raise NotImplementedError
