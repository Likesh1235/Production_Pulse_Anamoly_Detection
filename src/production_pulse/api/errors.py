"""Uniform JSON error responses that never expose stack traces or file paths.

Every error body has the shape::

    {"error": {"code": "invalid_features", "message": "...", "details": [...]}}
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..logging_config import get_logger

log = get_logger("api.errors")


class ApiError(Exception):
    """An expected error with a stable machine-readable code."""

    def __init__(self, status_code: int, code: str, message: str, details: Optional[Any] = None) -> None:
        super().__init__(message)
        self.status_code, self.code, self.message, self.details = status_code, code, message, details


def error_body(code: str, message: str, details: Optional[Any] = None) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        body["details"] = details
    return {"error": body}


def _clean_validation_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    out = []
    for err in exc.errors()[:50]:
        loc = [str(p) for p in err.get("loc", ())]
        msg = str(err.get("msg", "invalid value"))
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        out.append({"location": loc, "message": msg, "type": str(err.get("type", "value_error"))})
    return out


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=error_body(exc.code, exc.message, exc.details))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_body("validation_error", "The request is invalid.", _clean_validation_errors(exc)),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {404: "not_found", 405: "method_not_allowed"}
        message = "Resource not found." if exc.status_code == 404 else str(exc.detail)
        return JSONResponse(status_code=exc.status_code, content=error_body(codes.get(exc.status_code, "http_error"), message))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error while serving a request")  # full detail stays in the server log
        return JSONResponse(status_code=500, content=error_body("internal_error", "Internal server error."))
