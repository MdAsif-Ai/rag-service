import os
import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Test environment defaults
# ---------------------------------------------------------------------------
# These are only fallback values for unit/API tests.
# Real integration credentials should come from the environment/.env.
os.environ.setdefault("RAG_SERVICE_API_KEY", "test-api-key")

os.environ.setdefault(
    "SUPABASE_URL",
    "http://localhost:54321",
)

os.environ.setdefault(
    "SUPABASE_SERVICE_ROLE_KEY",
    "test-supabase-key",
)

os.environ.setdefault(
    "QDRANT_URL",
    "http://localhost:6333",
)

# IMPORTANT:
# Do NOT provide a fake Qdrant API key here.
#
# If QDRANT_API_KEY already exists in the environment, the application
# will use it for integration tests.
#
# Unit tests should use mock_settings() below instead.


# ---------------------------------------------------------------------------
# Make application importable
# ---------------------------------------------------------------------------
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    ),
)


# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires RUN_INTEGRATION_TESTS=true",
    )

    config.addinivalue_line(
        "markers",
        "e2e: requires RUN_E2E_TESTS=true",
    )


# ---------------------------------------------------------------------------
# Skip integration/E2E tests unless explicitly enabled
# ---------------------------------------------------------------------------
def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_INTEGRATION_TESTS") != "true":
        skip_integration = pytest.mark.skip(
            reason="Needs RUN_INTEGRATION_TESTS=true"
        )

        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    if os.getenv("RUN_E2E_TESTS") != "true":
        skip_e2e = pytest.mark.skip(
            reason="Needs RUN_E2E_TESTS=true"
        )

        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)


# ---------------------------------------------------------------------------
# Mock application settings for unit tests
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_settings():
    settings = MagicMock()

    settings.APP_NAME = "Test RAG"
    settings.APP_ENV = "test"

    # Qdrant
    settings.QDRANT_URL = "http://localhost:6333"
    settings.QDRANT_API_KEY = "test-key"
    settings.QDRANT_COLLECTION = "test_collection"
    settings.QDRANT_TIMEOUT = 10

    # Reranker
    settings.RERANKER_MODEL = "test-reranker"
    settings.RERANKER_DEVICE = "cpu"
    settings.RERANK_BATCH_SIZE = 2
    settings.RERANK_MAX_CANDIDATES = 5
    settings.RERANK_FINAL_TOP_K = 3

    # Chunking
    settings.CHUNK_SIZE_TOKENS = 100
    settings.CHUNK_OVERLAP_TOKENS = 10

    return settings