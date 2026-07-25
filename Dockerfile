FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# .[ml] pulls the heavy, model-backed deps (sentence-transformers, faster-whisper,
# google-generativeai) that CI intentionally skips — every import of them is lazy,
# so tests don't need this image, but real transcription/embedding/synthesis do.
RUN pip install --no-cache-dir ".[ml]"

EXPOSE 8000

CMD ["uvicorn", "videorag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
