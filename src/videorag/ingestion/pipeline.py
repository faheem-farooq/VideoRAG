import tempfile
from dataclasses import dataclass
from pathlib import Path

from videorag.ingestion.audio import extract_audio
from videorag.ingestion.chunking import Chunk, chunk_transcript
from videorag.ingestion.embedding import Embedder
from videorag.ingestion.transcription.base import TranscriptionBackend
from videorag.retrieval.store import VectorStore


@dataclass(frozen=True)
class IngestionResult:
    video_id: str
    num_chunks: int
    chunks: list[Chunk]


def ingest_video(
    video_id: str,
    video_path: Path,
    transcription_backend: TranscriptionBackend,
    embedder: Embedder,
    store: VectorStore,
    language: str = "en",
) -> IngestionResult:
    """Run the full offline pipeline for one video: extract audio, transcribe,
    chunk, embed, and persist to the vector store. Raises on failure rather than
    swallowing errors — the caller (a background job) is responsible for
    recording failure status.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = extract_audio(video_path, Path(tmp_dir) / f"{video_id}.wav")
        segments = transcription_backend.transcribe(audio_path)

    if not segments:
        raise ValueError(f"No speech detected in video {video_id!r}")

    chunks = chunk_transcript(segments)
    embeddings = embedder.embed([c.text for c in chunks])
    store.add_chunks(video_id=video_id, chunks=chunks, embeddings=embeddings, language=language)

    return IngestionResult(video_id=video_id, num_chunks=len(chunks), chunks=chunks)
