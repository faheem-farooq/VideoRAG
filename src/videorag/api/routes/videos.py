import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from videorag.api.deps import get_state
from videorag.api.jobs import JobStatus
from videorag.api.schemas import VideoIngestResponse, VideoListResponse, VideoStatusResponse
from videorag.api.state import AppState
from videorag.ingestion.pipeline import ingest_video

logger = logging.getLogger("videorag.videos")

router = APIRouter(prefix="/videos", tags=["videos"])


def _run_ingestion(state: AppState, video_id: str, video_path: Path, language: str) -> None:
    state.job_store.update_status(video_id, JobStatus.PROCESSING)
    try:
        result = ingest_video(
            video_id=video_id,
            video_path=video_path,
            transcription_backend=state.transcription_backend,
            embedder=state.embedder,
            store=state.store,
            language=language,
        )
        state.job_store.update_status(video_id, JobStatus.READY, num_chunks=result.num_chunks)
    except Exception as exc:  # noqa: BLE001 — must not crash the background task runner
        logger.exception("Ingestion failed for video_id=%s", video_id)
        state.job_store.update_status(video_id, JobStatus.FAILED, error=str(exc))


@router.post("", response_model=VideoIngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_video(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    language: str = Form(default="en"),
    state: AppState = Depends(get_state),
) -> VideoIngestResponse:
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected a video file, got content-type {file.content_type!r}",
        )

    video_id = uuid.uuid4().hex
    suffix = Path(file.filename or "").suffix or ".mp4"
    video_path = state.settings.upload_dir / f"{video_id}{suffix}"

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty"
        )
    video_path.write_bytes(contents)

    state.job_store.create(
        video_id=video_id,
        filename=file.filename or video_path.name,
        language=language,
        stored_filename=video_path.name,
    )
    background_tasks.add_task(_run_ingestion, state, video_id, video_path, language)

    return VideoIngestResponse(video_id=video_id, status=JobStatus.PENDING)


@router.get("/{video_id}/status", response_model=VideoStatusResponse)
async def get_video_status(
    video_id: str, state: AppState = Depends(get_state)
) -> VideoStatusResponse:
    job = state.job_store.get(video_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No video with id {video_id!r}"
        )
    return VideoStatusResponse(**job.model_dump())


@router.get("", response_model=VideoListResponse)
async def list_videos(state: AppState = Depends(get_state)) -> VideoListResponse:
    jobs = state.job_store.list_all()
    return VideoListResponse(videos=[VideoStatusResponse(**job.model_dump()) for job in jobs])


@router.get("/{video_id}/file")
async def get_video_file(video_id: str, state: AppState = Depends(get_state)) -> FileResponse:
    job = state.job_store.get(video_id)
    if job is None or not job.stored_filename:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No video with id {video_id!r}"
        )

    video_path = state.settings.upload_dir / job.stored_filename
    if not video_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Video file not found on disk"
        )

    return FileResponse(video_path, filename=job.filename)
