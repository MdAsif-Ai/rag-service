# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.12-slim AS builder

# Install build dependencies required by some ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official standalone installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Increase UV timeout to 300 seconds to prevent network timeouts on large ML packages
ENV UV_HTTP_TIMEOUT=300

# Copy only dependency manifests first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Create a virtual environment and install dependencies from lockfile
RUN uv venv /app/.venv && \
    uv sync --frozen --no-dev --no-install-project

# ==========================================
# Stage 2: Runtime
# ==========================================
FROM python:3.12-slim AS runtime

# Install tini, curl, AND system dependencies for Docling/OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    curl \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --chown=appuser:appuser --from=builder /app/.venv /app/.venv

# Activate the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set HuggingFace cache directory to a path we can mount
ENV HF_HOME=/app/hf_cache

# Copy application code
COPY --chown=appuser:appuser . /app

# Create cache directory and give ownership to appuser
RUN mkdir -p /app/hf_cache && chown -R appuser:appuser /app/hf_cache

USER appuser

ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command (overridden by docker-compose for the worker)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]