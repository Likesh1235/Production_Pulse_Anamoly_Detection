# Production Pulse API (FastAPI + TensorFlow). Build from the project folder:
#   docker build -t production-pulse-api .
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PP_API_HOST=0.0.0.0 PRODUCTION_PULSE_ROOT=/app
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY artifacts/models/autoencoder_v1 ./artifacts/models/autoencoder_v1
COPY reports/evaluation_metrics.json ./reports/evaluation_metrics.json
RUN pip install . && mkdir -p data/api logs
# Hosts such as Render/Fly/Railway provide $PORT; default 8000 locally.
CMD ["sh", "-c", "PP_API_PORT=${PORT:-8000} exec python -m production_pulse.api"]
