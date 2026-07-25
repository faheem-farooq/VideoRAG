import hashlib
import logging
import time

from fastapi import APIRouter, Depends, Request

from videorag.api.deps import get_state
from videorag.api.limiter import limiter
from videorag.api.schemas import QueryRequest, QueryResponse, SegmentResult
from videorag.api.state import AppState
from videorag.core.config import get_settings

logger = logging.getLogger("videorag.query")

router = APIRouter(tags=["query"])


def _cache_key(payload: QueryRequest) -> str:
    raw = (
        f"{payload.query.strip().lower()}|{payload.video_id}|{payload.top_k}|"
        f"{payload.rerank}|{payload.synthesize_answer}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()


@router.post("/query", response_model=QueryResponse)
@limiter.limit(get_settings().rate_limit_query)
async def query_videos(
    request: Request,
    payload: QueryRequest,
    state: AppState = Depends(get_state),
) -> QueryResponse:
    start = time.perf_counter()
    cache_key = _cache_key(payload)

    cached = state.query_cache.get(cache_key)
    if cached is not None:
        logger.info("Cache hit for query cache_key=%s", cache_key[:8])
        return cached.model_copy(
            update={"cached": True, "latency_ms": (time.perf_counter() - start) * 1000}
        )

    query_embedding = state.embedder.embed_one(payload.query)
    segments = state.store.query(query_embedding, top_k=payload.top_k, video_id=payload.video_id)

    if payload.rerank and segments:
        segments = state.reranker.rerank(payload.query, segments, top_k=payload.top_k)

    answer = None
    if payload.synthesize_answer:
        answer = state.synthesizer.synthesize(payload.query, segments)

    response = QueryResponse(
        query=payload.query,
        segments=[
            SegmentResult(video_id=s.video_id, start=s.start, end=s.end, text=s.text, score=s.score)
            for s in segments
        ],
        answer=answer,
        cached=False,
        latency_ms=(time.perf_counter() - start) * 1000,
    )
    state.query_cache[cache_key] = response
    return response
