import pytest
from uuid import uuid4
from app.vectorstore.qdrant import get_qdrant_repository

@pytest.fixture(scope="module")
def qdrant_repo():
    # Use the factory which reads QDRANT_URL and QDRANT_API_KEY from .env
    repo = get_qdrant_repository()
    if not repo.collection_exists():
        repo.create_collection()
    yield repo
    # Cleanup after module
    repo.client.delete_collection(repo.collection_name)

@pytest.mark.integration
def test_qdrant_upsert_and_search(qdrant_repo):
    doc_id = uuid4()
    chunks = [{
        "chunk_index": 0,
        "content": "Test chunk for integration",
        "dense_vector": [0.1] * 1024,  # BGE-M3 size
        "sparse_vector": {1: 0.5, 2: 0.2}
    }]
    
    qdrant_repo.upsert_points(doc_id, "course-int", "test.pdf", chunks)
    
    # Dense search
    results = qdrant_repo.search_dense([0.1] * 1024, ["course-int"], top_k=1)
    assert len(results) == 1
    assert results[0]["metadata"]["document_id"] == str(doc_id)
    
    # Sparse search
    results = qdrant_repo.search_sparse({1: 0.5}, ["course-int"], top_k=1)
    assert len(results) == 1
    
    # Cleanup doc
    qdrant_repo.delete_document(doc_id)