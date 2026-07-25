from tests.integration.conftest import FakeTranscriptionBackend


def test_create_video_rejects_non_video_content_type(client):
    response = client.post(
        "/videos",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_create_video_rejects_empty_file(client):
    response = client.post(
        "/videos",
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )
    assert response.status_code == 400


def test_status_for_unknown_video_is_404(client):
    response = client.get("/videos/does-not-exist/status")
    assert response.status_code == 404


def test_create_video_ingests_successfully_and_status_becomes_ready(client):
    response = client.post(
        "/videos",
        files={"file": ("cats.mp4", b"fake video bytes", "video/mp4")},
        data={"language": "en"},
    )
    assert response.status_code == 202
    video_id = response.json()["video_id"]
    assert response.json()["status"] == "pending"

    status_response = client.get(f"/videos/{video_id}/status")
    assert status_response.status_code == 200
    body = status_response.json()
    assert body["status"] == "ready"
    assert body["num_chunks"] > 0
    assert body["error"] is None


def test_list_videos_includes_created_video(client):
    create_response = client.post(
        "/videos",
        files={"file": ("cats.mp4", b"fake video bytes", "video/mp4")},
    )
    video_id = create_response.json()["video_id"]

    list_response = client.get("/videos")
    assert list_response.status_code == 200
    ids = [v["video_id"] for v in list_response.json()["videos"]]
    assert video_id in ids


def test_get_video_file_returns_uploaded_bytes(client):
    create_response = client.post(
        "/videos",
        files={"file": ("cats.mp4", b"fake video bytes", "video/mp4")},
    )
    video_id = create_response.json()["video_id"]

    file_response = client.get(f"/videos/{video_id}/file")
    assert file_response.status_code == 200
    assert file_response.content == b"fake video bytes"


def test_get_video_file_for_unknown_video_is_404(client):
    response = client.get("/videos/does-not-exist/file")
    assert response.status_code == 404


def test_ingestion_failure_marks_job_failed(client, app_state):
    app_state.transcription_backend = FakeTranscriptionBackend(should_fail=True)

    response = client.post(
        "/videos",
        files={"file": ("broken.mp4", b"fake video bytes", "video/mp4")},
    )
    video_id = response.json()["video_id"]

    status_response = client.get(f"/videos/{video_id}/status")
    body = status_response.json()
    assert body["status"] == "failed"
    assert "fake transcription failure" in body["error"]
