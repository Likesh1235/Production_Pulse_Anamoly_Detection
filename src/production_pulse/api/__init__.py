"""Production Pulse REST API (FastAPI).

``create_app`` and ``ApiServices`` are imported lazily, so the FastAPI-free parts of
this package (``settings``, ``store``) can be used without FastAPI installed.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ApiServices", "create_app"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import app as _app

        return getattr(_app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
