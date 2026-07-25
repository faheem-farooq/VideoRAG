from pathlib import Path

from videorag.ingestion.transcription.base import TranscriptionBackend, TranscriptSegment


class FasterWhisperBackend(TranscriptionBackend):
    """Local CPU-friendly transcription via faster-whisper (CTranslate2 Whisper)."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_size, device=self._device, compute_type=self._compute_type
            )
        return self._model

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        model = self._load_model()
        segments, _info = model.transcribe(str(audio_path), vad_filter=True)
        return [
            TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip())
            for seg in segments
            if seg.text.strip()
        ]
