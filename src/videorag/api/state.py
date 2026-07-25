from dataclasses import dataclass, field

from cachetools import TTLCache

from videorag.api.jobs import JobStore
from videorag.core.config import Settings
from videorag.ingestion.embedding import Embedder
from videorag.ingestion.transcription.base import TranscriptionBackend
from videorag.llm.gemini import GeminiSynthesizer
from videorag.retrieval.rerank import CrossEncoderReranker
from videorag.retrieval.store import VectorStore


@dataclass
class AppState:
    settings: Settings
    embedder: Embedder
    transcription_backend: TranscriptionBackend
    store: VectorStore
    job_store: JobStore
    synthesizer: GeminiSynthesizer
    reranker: CrossEncoderReranker = field(default_factory=CrossEncoderReranker)
    query_cache: TTLCache = field(default_factory=lambda: TTLCache(maxsize=1024, ttl=300))


def build_app_state(settings: Settings) -> AppState:
    from videorag.ingestion.transcription import get_transcription_backend
    from videorag.retrieval.store import get_http_client

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.job_db_path.parent.mkdir(parents=True, exist_ok=True)

    client = get_http_client(settings.chroma_host, settings.chroma_port)
    store = VectorStore(client, collection_name=settings.chroma_collection)

    transcription_backend = get_transcription_backend(
        settings.transcription_backend,
        model_size=settings.faster_whisper_model_size,
        device=settings.faster_whisper_device,
        compute_type=settings.faster_whisper_compute_type,
    )

    return AppState(
        settings=settings,
        embedder=Embedder(settings.embedding_model),
        transcription_backend=transcription_backend,
        store=store,
        job_store=JobStore(settings.job_db_path),
        synthesizer=GeminiSynthesizer(settings.gemini_api_key, settings.gemini_model),
        query_cache=TTLCache(maxsize=1024, ttl=settings.query_cache_ttl_seconds),
    )
