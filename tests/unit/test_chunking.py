from videorag.ingestion.chunking import chunk_transcript
from videorag.ingestion.transcription.base import TranscriptSegment


def test_short_segments_merge_into_one_chunk():
    segments = [
        TranscriptSegment(start=0.0, end=2.0, text="Hello there."),
        TranscriptSegment(start=2.0, end=4.0, text="This is a test."),
    ]
    chunks = chunk_transcript(segments, max_chars=500, max_duration_seconds=30.0)
    assert len(chunks) == 1
    assert chunks[0].start == 0.0
    assert chunks[0].end == 4.0
    assert chunks[0].text == "Hello there. This is a test."


def test_chunk_splits_on_duration_limit():
    segments = [
        TranscriptSegment(start=0.0, end=20.0, text="First long segment."),
        TranscriptSegment(start=20.0, end=40.0, text="Second long segment."),
    ]
    chunks = chunk_transcript(segments, max_chars=500, max_duration_seconds=30.0)
    assert len(chunks) == 2
    assert chunks[0].text == "First long segment."
    assert chunks[1].text == "Second long segment."


def test_chunk_splits_on_char_limit_without_cutting_a_segment_midway():
    segments = [
        TranscriptSegment(start=0.0, end=1.0, text="A" * 300),
        TranscriptSegment(start=1.0, end=2.0, text="B" * 300),
    ]
    chunks = chunk_transcript(segments, max_chars=500, max_duration_seconds=1000.0)
    assert len(chunks) == 2
    assert chunks[0].text == "A" * 300
    assert chunks[1].text == "B" * 300


def test_oversized_single_segment_is_split_on_sentence_boundaries():
    long_text = "First sentence here. Second sentence here. Third sentence here."
    segments = [TranscriptSegment(start=0.0, end=9.0, text=long_text)]
    chunks = chunk_transcript(segments, max_chars=20, max_duration_seconds=1000.0)

    assert len(chunks) >= 2
    reconstructed = " ".join(c.text for c in chunks)
    for sentence in ["First sentence here.", "Second sentence here.", "Third sentence here."]:
        assert sentence in reconstructed

    for chunk in chunks:
        assert 0.0 <= chunk.start <= chunk.end <= 9.0


def test_empty_input_returns_no_chunks():
    assert chunk_transcript([]) == []
