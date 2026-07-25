import re
from dataclasses import dataclass

from videorag.ingestion.transcription.base import TranscriptSegment

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(frozen=True)
class Chunk:
    start: float
    end: float
    text: str


def _split_long_segment(segment: TranscriptSegment, max_chars: int) -> list[TranscriptSegment]:
    """Fallback for a single Whisper segment that already exceeds max_chars.

    Splits on sentence punctuation and interpolates timestamps proportionally by
    character offset within the segment, since Whisper doesn't give sub-segment
    timestamps.
    """
    sentences = [s for s in _SENTENCE_BOUNDARY.split(segment.text) if s.strip()]
    if len(sentences) <= 1:
        return [segment]

    duration = segment.end - segment.start
    total_chars = len(segment.text)
    pieces: list[TranscriptSegment] = []
    offset = 0
    for sentence in sentences:
        piece_start = (
            segment.start + duration * (offset / total_chars) if total_chars else segment.start
        )
        offset += len(sentence) + 1
        piece_end = (
            segment.start + duration * (min(offset, total_chars) / total_chars)
            if total_chars
            else segment.end
        )
        pieces.append(TranscriptSegment(start=piece_start, end=piece_end, text=sentence.strip()))
    return pieces


def chunk_transcript(
    segments: list[TranscriptSegment],
    max_chars: int = 500,
    max_duration_seconds: float = 30.0,
) -> list[Chunk]:
    """Group timestamped transcript segments into retrieval-sized chunks.

    Whisper segments are already sentence/phrase-bounded, so this only ever joins
    whole segments together (or, for an oversized single segment, splits at
    sentence punctuation) — text is never cut mid-sentence.
    """
    normalized: list[TranscriptSegment] = []
    for seg in segments:
        if len(seg.text) > max_chars:
            normalized.extend(_split_long_segment(seg, max_chars))
        else:
            normalized.append(seg)

    chunks: list[Chunk] = []
    current: list[TranscriptSegment] = []

    def flush() -> None:
        if not current:
            return
        text = " ".join(s.text for s in current).strip()
        chunks.append(Chunk(start=current[0].start, end=current[-1].end, text=text))

    for seg in normalized:
        if current:
            prospective_text_len = sum(len(s.text) for s in current) + len(seg.text) + 1
            prospective_duration = seg.end - current[0].start
            if prospective_text_len > max_chars or prospective_duration > max_duration_seconds:
                flush()
                current = []
        current.append(seg)

    flush()
    return chunks
