from datetime import datetime

from pydantic import BaseModel, Field

from videorag.api.jobs import JobStatus


class VideoIngestResponse(BaseModel):
    video_id: str
    status: JobStatus


class VideoStatusResponse(BaseModel):
    video_id: str
    filename: str
    status: JobStatus
    num_chunks: int
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class VideoListResponse(BaseModel):
    videos: list[VideoStatusResponse]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    video_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = False
    synthesize_answer: bool = True


class SegmentResult(BaseModel):
    video_id: str
    start: float
    end: float
    text: str
    score: float


class QueryResponse(BaseModel):
    query: str
    segments: list[SegmentResult]
    answer: str | None = None
    cached: bool = False
    latency_ms: float
