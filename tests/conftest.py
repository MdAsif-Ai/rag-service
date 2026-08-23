import os
import sys
import pytest
from unittest.mock import MagicMock

# Ensure app is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Environment markers
def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires RUN_INTEGRATION_TESTS=true")
    config.addinivalue_line("markers", "e2e: requires RUN_E2E_TESTS=true")

# Skip integration tests unless explicitly enabled
def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_INTEGRATION_TESTS") != "true":
        skip_integration = pytest.mark.skip(reason="Needs RUN_INTEGRATION_TESTS=true")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    if os.getenv("RUN_E2E_TESTS") != "true":
        skip_e2e = pytest.mark.skip(reason="Needs RUN_E2E_TESTS=true")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)

@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.APP_NAME = "Test RAG"
    settings.APP_ENV = "test"
    settings.QDRANT_URL = "http://localhost:6333"
    settings.QDRANT_API_KEY = "test-key"
    settings.QDRANT_COLLECTION = "test_collection"
    settings.RERANKER_MODEL = "test-reranker"
    settings.RERANKER_DEVICE = "cpu"
    settings.RERANK_BATCH_SIZE = 2
    settings.RERANK_MAX_CANDIDATES = 5
    settings.RERANK_FINAL_TOP_K = 3
    settings.CHUNK_SIZE_TOKENS = 100
    settings.CHUNK_OVERLAP_TOKENS = 10
    return settings