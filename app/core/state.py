from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.retrieval.pipeline import RetrievalPipeline

class AppState:
    # Explicitly define the attribute so it always exists
    pipeline: Optional["RetrievalPipeline"] = None

app_state = AppState()