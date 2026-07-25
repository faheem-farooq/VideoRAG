import uuid

import pytest

from videorag.ingestion.chunking import Chunk
from videorag.retrieval.store import VectorStore, get_ephemeral_client


def _fake_embedding(seed: float) -> list[float]:
    # Deterministic 8-dim vectors so cosine similarity ordering is predictable.
    return [seed, 1.0 - seed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def store() -> VectorStore:
    # chromadb's EphemeralClient() shares process-global state keyed by settings,
    # so a unique collection name per test is what actually gives isolation.
    client = get_ephemeral_client()
    return VectorStore(client, collection_name=f"test_segments_{uuid.uuid4().hex}")


def test_add_and_query_returns_matching_video(store: VectorStore):
    chunks = [Chunk(start=0.0, end=5.0, text="intro segment")]
    store.add_chunks("video-a", chunks, [_fake_embedding(1.0)], language="en")

    results = store.query(_fake_embedding(1.0), top_k=1)

    assert len(results) == 1
    assert results[0].video_id == "video-a"
    assert results[0].text == "intro segment"
    assert results[0].start == 0.0
    assert results[0].end == 5.0


def test_query_filters_by_video_id(store: VectorStore):
    store.add_chunks("video-a", [Chunk(0.0, 5.0, "from a")], [_fake_embedding(1.0)])
    store.add_chunks("video-b", [Chunk(0.0, 5.0, "from b")], [_fake_embedding(1.0)])

    results = store.query(_fake_embedding(1.0), top_k=5, video_id="video-b")

    assert len(results) == 1
    assert results[0].video_id == "video-b"


def test_list_video_ids_returns_unique_sorted_ids(store: VectorStore):
    store.add_chunks("video-b", [Chunk(0.0, 5.0, "b")], [_fake_embedding(1.0)])
    store.add_chunks("video-a", [Chunk(0.0, 5.0, "a")], [_fake_embedding(0.0)])

    assert store.list_video_ids() == ["video-a", "video-b"]


def test_delete_video_removes_its_chunks(store: VectorStore):
    store.add_chunks("video-a", [Chunk(0.0, 5.0, "a")], [_fake_embedding(1.0)])
    store.delete_video("video-a")

    assert store.list_video_ids() == []


def test_add_chunks_with_empty_list_is_a_noop(store: VectorStore):
    store.add_chunks("video-a", [], [])
    assert store.list_video_ids() == []
