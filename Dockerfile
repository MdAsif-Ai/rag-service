# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.12-slim AS builder

# Install build dependencies required by some ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy only dependency manifests first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Create a virtual environment and install dependencies
# --frozen ensures deterministic installation from uv.lock
# --no-dev excludes pytest, ruff, etc.
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python --frozen --no-dev .

# ==========================================
# Stage 2: Runtime
# ==========================================
FROM python:3.12-slim AS runtime

# Install tini for proper signal handling (prevents zombie processes, 
# ensures Celery workers shut down gracefully) and curl for healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Set working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --chown=appuser:appuser --from=builder /app/.venv /app/.venv

# Activate the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY --chown=appuser:appuser . /app

# Switch to non-root user
USER appuser

# Use tini as the entrypoint
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command (can be overridden by docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]