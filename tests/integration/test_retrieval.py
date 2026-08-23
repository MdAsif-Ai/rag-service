import pytest
# This test would wire up real Qdrant client, real embedding model, etc.
# It is intentionally left skeletal as it requires heavy ML models to run.

@pytest.mark.integration
def test_full_retrieval_pipeline_real_models():
    # pytest.skip("Requires real BGE-M3 model downloaded locally")
    # setup real pipeline
    # query = "What is Python?"
    # response = await pipeline.retrieve(query, ["course-1"], 5)
    # assert response.final_count > 0
    pass