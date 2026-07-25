import uuid

import pytest
from fastapi.testclient import TestClient

from videorag.api.jobs import JobStore
from videorag.api.main import create_app
from videorag.api.state import AppState
from videorag.core.config import Settings
from videorag.ingestion.transcription.base import TranscriptionBackend, TranscriptSegment
from videorag.llm.gemini import GeminiSynthesizer
from videorag.retrieval.store import VectorStore, get_ephemeral_client

DEFAULT_SEGMENTS = [
    TranscriptSegment(start=0.0, end=3.0, text="Hello, this is a test video about cats."),
    TranscriptSegment(start=3.0, end=6.0, text="Cats are wonderful independent pets."),
    TranscriptSegment(start=6.0, end=9.0, text="Dogs on the other hand love company."),
]


class FakeTranscriptionBackend(TranscriptionBackend):
    def __init__(self, segments=None, should_fail=False):
        self._segments = segments if segments is not None else DEFAULT_SEGMENTS
        self._should_fail = should_fail

    def transcribe(self, audio_path):
        if self._should_fail:
            raise RuntimeError("fake transcription failure")
        return self._segments


class FakeEmbedder:
    """Deterministic hashing-based fake embedder — no model download needed.

    Consistent within a single test process (Python's str hashing is randomized
    per-process, not per-call), which is all these tests need: ingest then query
    happen in the same process.
    """

    def __init__(self, model_name: str = "fake") -> None:
        self._dim = 16

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for word in text.lower().split():
            vec[hash(word) % self._dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


@pytest.fixture(autouse=True)
def _fake_audio_extraction(monkeypatch):
    """Integration tests exercise the API/orchestration layer, not ffmpeg itself
    (which the ingestion unit tests don't cover either — it's a thin subprocess
    wrapper). Uploaded test files are not real video, so real ffmpeg would fail.
    """

    def _fake_extract_audio(video_path, output_path, sample_rate=16000):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"")
        return output_path

    monkeypatch.setattr("videorag.ingestion.pipeline.extract_audio", _fake_extract_audio)


@pytest.fixture
def app_state(tmp_path) -> AppState:
    settings = Settings(
        upload_dir=tmp_path / "uploads",
        job_db_path=tmp_path / "jobs.db",
        gemini_api_key="",
        chroma_collection=f"test_{uuid.uuid4().hex}",
    )
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    client = get_ephemeral_client()
    store = VectorStore(client, collection_name=settings.chroma_collection)

    return AppState(
        settings=settings,
        embedder=FakeEmbedder(),
        transcription_backend=FakeTranscriptionBackend(),
        store=store,
        job_store=JobStore(settings.job_db_path),
        synthesizer=GeminiSynthesizer(api_key=""),
    )


@pytest.fixture
def client(app_state: AppState):
    app = create_app(state=app_state)
    with TestClient(app) as test_client:
        yield test_client
