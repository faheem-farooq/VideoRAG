import contextvars
import json
import logging
import re
import sys
import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([^\s\"',&]+)"),
    re.compile(r"(?i)(gemini_api_key\s*[:=]\s*)([^\s\"',&]+)"),
    re.compile(r"(AIza[0-9A-Za-z\-_]{20,})"),
]


def redact_secrets(message: str) -> str:
    redacted = message
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(
            lambda m: m.group(1) + "***REDACTED***" if m.lastindex else "***REDACTED***", redacted
        )
    return redacted


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_secrets(record.getMessage()),
            "request_id": request_id_var.get(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        logging.getLogger("videorag.access").info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            getattr(response, "status_code", "?"),
            duration_ms,
        )
        return response
