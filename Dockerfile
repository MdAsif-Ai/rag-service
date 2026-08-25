FROM python:3.12-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv venv /app/.venv \
    && uv sync --frozen --no-dev --no-install-project


FROM python:3.12-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends tini curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system appuser \
    && useradd --system \
       --gid appuser \
       --create-home \
       --home-dir /app \
       --shell /usr/sbin/nologin \
       appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME="/app/hf_cache"
ENV TRANSFORMERS_CACHE="/app/hf_cache"
ENV HF_HUB_CACHE="/app/hf_cache/hub"

COPY --chown=appuser:appuser . /app

RUN mkdir -p /app/hf_cache \
    && chown -R appuser:appuser /app/hf_cache

USER appuser

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]