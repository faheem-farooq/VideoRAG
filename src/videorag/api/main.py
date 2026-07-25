import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from videorag.api.limiter import limiter
from videorag.api.routes import query as query_routes
from videorag.api.routes import videos as video_routes
from videorag.api.state import AppState, build_app_state
from videorag.core.config import Settings, get_settings
from videorag.core.logging import RequestIdMiddleware, configure_logging

logger = logging.getLogger("videorag.api")


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Rate limit exceeded, please slow down."},
    )


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error while processing %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


def create_app(state: AppState | None = None, settings: Settings | None = None) -> FastAPI:
    """App factory. Pass a pre-built AppState (with mocked Whisper/Gemini/Chroma)
    for tests; production entrypoint builds real dependencies lazily on startup.
    """
    resolved_settings = settings or (state.settings if state else get_settings())
    configure_logging(resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.videorag = state or build_app_state(resolved_settings)
        yield

    app = FastAPI(
        title="VideoRAG",
        version="0.1.0",
        description=(
            "Multilingual query-driven video segment retrieval. Cross-lingual "
            "search happens at the embedding level; the LLM is only used for "
            "final answer synthesis, not per-query translation."
        ),
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)

    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(video_routes.router)
    app.include_router(query_routes.router)

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
