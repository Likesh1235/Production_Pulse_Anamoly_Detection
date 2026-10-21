"""FastAPI application factory for Production Pulse.

``create_app()`` builds an isolated app (handy for tests). The model is loaded in
the lifespan handler at startup - never trained here - and the API starts even
when no model exists: ``/health`` keeps working and ``/predict`` returns 503.
Run it with Uvicorn (see docs/API.md), e.g.::

    python -m uvicorn production_pulse.api.main:app --app-dir src
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..config import ApiConfig, ProjectConfig, load_config
from ..logging_config import get_logger
from ..paths import ProjectPaths, UnsafeOutputPathError, get_paths
from .errors import install_error_handlers
from .routes import router
from .service import ModelService
from .settings import apply_env_overrides, read_admin_token
from .store import PredictionStore, StoreError

log = get_logger("api.app")

DB_UNAVAILABLE_REASON = "The prediction history database is unavailable. See the server log for details."


@dataclass
class ApiServices:
    """Everything a request handler needs; stored on ``app.state.services``."""

    api: ApiConfig
    paths: ProjectPaths
    model: ModelService
    admin_token: Optional[str]
    store: Optional[PredictionStore] = None
    store_status: str = "not_initialised"

    def start(self, *, load_model: bool) -> None:
        try:
            self.store = PredictionStore(self.api.database_path, self.paths, store_inputs=self.api.store_inputs)
            self.store_status = "ready"
        except (UnsafeOutputPathError, StoreError, OSError) as exc:
            log.error("History database disabled: %s: %s", type(exc).__name__, exc)
            self.store, self.store_status = None, "unavailable"
        if load_model:
            self.model.load()

    def store_component(self) -> dict[str, Any]:
        ok = self.store is not None and self.store.ping()
        return {"available": ok, "status": "ready" if ok else "unavailable", "reason": None if ok else DB_UNAVAILABLE_REASON}


def create_app(
    *,
    config: Optional[ProjectConfig] = None,
    paths: Optional[ProjectPaths] = None,
    environ: Optional[Mapping[str, str]] = None,
    model_loader: Optional[Callable[..., Any]] = None,
    load_model_on_startup: bool = True,
) -> FastAPI:
    """Create the API. ``environ`` defaults to ``os.environ`` (PP_API_* overrides, admin token)."""
    environ = os.environ if environ is None else environ
    paths = paths if paths is not None else get_paths(config=config)
    config = config if config is not None else load_config(root=paths.root)
    api_cfg = apply_env_overrides(config.api, environ)
    services = ApiServices(
        api=api_cfg,
        paths=paths,
        model=ModelService(api_cfg.model_dir, paths, loader=model_loader, top_features=api_cfg.top_features),
        admin_token=read_admin_token(environ),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services.start(load_model=load_model_on_startup)
        yield

    app = FastAPI(
        title="Production Pulse API",
        version=__version__,
        description="Multivariate process anomaly detection (TEP, dense autoencoder).",
        lifespan=lifespan,
    )
    app.state.services = services
    if api_cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(api_cfg.cors_origins),   # explicit list only; '*' is rejected by ApiConfig
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-Admin-Token"],
        )
    install_error_handlers(app)
    app.include_router(router)
    return app
