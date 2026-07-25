from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # Transcription
    transcription_backend: str = "faster_whisper"
    faster_whisper_model_size: str = "small"
    faster_whisper_device: str = "cpu"
    faster_whisper_compute_type: str = "int8"

    # Embedding
    embedding_model: str = "sentence-transformers/LaBSE"

    # Chroma
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "video_segments"

    # Storage
    upload_dir: Path = Path("./data/uploads")
    job_db_path: Path = Path("./data/jobs.db")

    # API
    api_cors_origins: str = "http://localhost:3000"
    rate_limit_query: str = "30/minute"
    query_cache_ttl_seconds: int = 300
    log_level: str = "INFO"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.api_cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
