def _ingest_sample_video(client) -> str:
    response = client.post(
        "/videos",
        files={"file": ("cats.mp4", b"fake video bytes", "video/mp4")},
    )
    return response.json()["video_id"]


def test_query_returns_ranked_segments(client):
    video_id = _ingest_sample_video(client)

    response = client.post("/query", json={"query": "Tell me about cats", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["segments"], "expected at least one retrieved segment"
    assert all(s["video_id"] == video_id for s in body["segments"])
    assert body["cached"] is False
    assert body["latency_ms"] >= 0


def test_query_filters_by_video_id(client):
    _ingest_sample_video(client)

    response = client.post(
        "/query", json={"query": "cats", "video_id": "does-not-exist", "top_k": 3}
    )

    assert response.status_code == 200
    assert response.json()["segments"] == []


def test_query_without_gemini_key_returns_no_answer(client):
    _ingest_sample_video(client)

    response = client.post("/query", json={"query": "cats", "synthesize_answer": True})

    assert response.status_code == 200
    assert response.json()["answer"] is None


def test_repeated_identical_query_is_served_from_cache(client):
    _ingest_sample_video(client)

    first = client.post("/query", json={"query": "cats", "top_k": 2})
    second = client.post("/query", json={"query": "cats", "top_k": 2})

    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert second.json()["segments"] == first.json()["segments"]


def test_query_rejects_empty_query_string(client):
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422


def test_query_rejects_invalid_top_k(client):
    response = client.post("/query", json={"query": "cats", "top_k": 0})
    assert response.status_code == 422
