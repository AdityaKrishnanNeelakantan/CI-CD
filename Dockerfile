FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    SP_API_STATE_DIR=/app/.data/api \
    SP_PLATFORM_DB_PATH=/app/.data/platform.db \
    SP_STAGING_ROOT=/app/.data/staging

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
COPY streamlit_app.py ./

RUN uv sync --locked --no-dev --extra api --extra schema --extra parquet

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/.data \
    && chown -R appuser:appuser /app /opt/venv

USER appuser

EXPOSE 8000

CMD ["/opt/venv/bin/uvicorn", "synth_platform.interfaces.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
