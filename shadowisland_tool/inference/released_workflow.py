from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ._released_backend import ReleasedWorkflow


@lru_cache(maxsize=1)
def _load_workflow(model_dir: str) -> ReleasedWorkflow:
    return ReleasedWorkflow(Path(model_dir))


def predict_window_scores(records: list[object], model_dir: Path) -> list[dict[str, object]]:
    """Run the released manuscript prediction workflow.

    This is the only inference function used by the public pipeline.
    """
    return _load_workflow(str(Path(model_dir).resolve())).predict_records(records)

