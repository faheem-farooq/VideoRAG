import enum
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import Field, Session, SQLModel, create_engine, select


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class VideoJob(SQLModel, table=True):
    video_id: str = Field(primary_key=True)
    filename: str
    stored_filename: str = ""
    language: str = "en"
    status: JobStatus = JobStatus.PENDING
    num_chunks: int = 0
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JobStore:
    """SQLite-backed ingestion job tracker.

    A real message queue (Celery/Redis) would be the production-grade choice for
    background ingestion; this project runs a single API process, so a small SQL
    table updated from a FastAPI BackgroundTask is enough to track status durably
    across requests without adding an extra service. Documented as a known
    limitation (no distributed workers) rather than papered over.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(self._engine)

    def create(
        self, video_id: str, filename: str, language: str, stored_filename: str = ""
    ) -> VideoJob:
        job = VideoJob(
            video_id=video_id,
            filename=filename,
            language=language,
            stored_filename=stored_filename,
        )
        with Session(self._engine) as session:
            session.add(job)
            session.commit()
            session.refresh(job)
        return job

    def update_status(
        self,
        video_id: str,
        status: JobStatus,
        num_chunks: int | None = None,
        error: str | None = None,
    ) -> None:
        with Session(self._engine) as session:
            job = session.get(VideoJob, video_id)
            if job is None:
                return
            job.status = status
            job.updated_at = datetime.now(timezone.utc)
            if num_chunks is not None:
                job.num_chunks = num_chunks
            if error is not None:
                job.error = error
            session.add(job)
            session.commit()

    def get(self, video_id: str) -> VideoJob | None:
        with Session(self._engine) as session:
            return session.get(VideoJob, video_id)

    def list_all(self) -> list[VideoJob]:
        with Session(self._engine) as session:
            return list(session.exec(select(VideoJob).order_by(VideoJob.created_at.desc())))
