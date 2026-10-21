"""ASGI entry point: ``uvicorn production_pulse.api.main:app --app-dir src``.

Creating the app here is cheap; the model is loaded when Uvicorn starts the app.
"""

from .app import create_app

app = create_app()
