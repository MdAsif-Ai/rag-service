from typing import Any


class AppState:
    def __init__(self) -> None:
        self.retrieval_pipeline: Any = None


app_state = AppState()