# ============================================================
# Stage 1: Builder
# ============================================================

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ------------------------------------------------------------
# Build dependencies
# ------------------------------------------------------------

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*


# ------------------------------------------------------------
# Install uv
# ------------------------------------------------------------

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app


# ------------------------------------------------------------
# Copy dependency files first
# This allows Docker to cache dependency installation.
# ------------------------------------------------------------

COPY pyproject.toml uv.lock ./


# ------------------------------------------------------------
# Create virtual environment
# ------------------------------------------------------------

RUN uv venv /app/.venv


# ------------------------------------------------------------
# Install locked production dependencies
# ------------------------------------------------------------

RUN uv sync \
    --frozen \
    --no-dev \
    --no-install-project


# ============================================================
# Stage 2: Runtime
# ============================================================

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ------------------------------------------------------------
# Runtime packages
# ------------------------------------------------------------

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tini \
        curl \
    && rm -rf /var/lib/apt/lists/*


# ------------------------------------------------------------
# Create non-root application user
# ------------------------------------------------------------

RUN groupadd --system appuser \
    && useradd --system \
        --gid appuser \
        --create-home \
        --home-dir /app \
        --shell /usr/sbin/nologin \
        appuser


WORKDIR /app


# ------------------------------------------------------------
# Copy Python virtual environment
# ------------------------------------------------------------

COPY --from=builder \
    --chown=appuser:appuser \
    /app/.venv \
    /app/.venv


# ------------------------------------------------------------
# Use the virtual environment
# ------------------------------------------------------------

ENV PATH="/app/.venv/bin:$PATH"


# ------------------------------------------------------------
# Hugging Face cache
# ------------------------------------------------------------

ENV HF_HOME="/app/hf_cache"
ENV TRANSFORMERS_CACHE="/app/hf_cache"
ENV HF_HUB_CACHE="/app/hf_cache/hub"


# ------------------------------------------------------------
# Copy application
# ------------------------------------------------------------

COPY --chown=appuser:appuser . /app


# ------------------------------------------------------------
# Create Hugging Face cache directory
# ------------------------------------------------------------

RUN mkdir -p /app/hf_cache \
    && chown -R appuser:appuser /app/hf_cache


# ------------------------------------------------------------
# Run as non-root
# ------------------------------------------------------------

USER appuser


# ------------------------------------------------------------
# tini handles signals correctly for FastAPI/Celery
# ------------------------------------------------------------

ENTRYPOINT ["/usr/bin/tini", "--"]


# ------------------------------------------------------------
# Default API command
#
# docker-compose overrides this for rag-worker.
# ------------------------------------------------------------

CMD [
    "uvicorn",
    "app.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]