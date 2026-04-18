"""Structured JSON logging for API requests.

Each request gets a unique ``request_id`` propagated to logs. The middleware
emits one log line per request with: tenant_id, request_id, method, path,
status, latency_ms, char_count (for synthesize). Downstream log shipping
(CloudWatch, Datadog, Loki) consumes the JSON without further parsing.
"""

import json
import logging
import time
import uuid
from contextvars import ContextVar

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


REQUEST_ID_HEADER = "X-Request-ID"
_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_tenant_id_var: ContextVar[str] = ContextVar("tenant_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": _request_id_var.get(),
            "tenant_id": _tenant_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key in {"args", "msg", "levelname", "name", "exc_info", "exc_text", "stack_info"}:
                continue
            if key.startswith("_"):
                continue
            if key in payload:
                continue
            try:
                json.dumps(value)
            except TypeError:
                continue
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request_id and emit one structured access log per request."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._logger = logging.getLogger("api.access")

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        token_req = _request_id_var.set(request_id)
        token_ten = _tenant_id_var.set("-")
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._logger.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "latency_ms": round(latency_ms, 2),
                },
            )
            _request_id_var.reset(token_req)
            _tenant_id_var.reset(token_ten)


def bind_tenant(tenant_id: str) -> None:
    """Bind the resolved tenant to the current request's log context."""
    _tenant_id_var.set(tenant_id)
