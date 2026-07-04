from __future__ import annotations

import hashlib
from pathlib import Path


REQUIRED_MODEL_FILES = [
    "paper_model_v1_encoder.pth",
    "paper_model_v1_classifier.pth",
    "paper_model_v1_decoder.pt",
    "paper_model_v1_feature_scaler.pkl",
    "paper_model_v1_feature_model.pkl",
    "paper_model_v1_calibration.bin",
]


def check_model_bundle(model_dir: Path | str) -> dict[str, object]:
    model_dir = Path(model_dir)
    files = []
    ready = True
    for name in REQUIRED_MODEL_FILES:
        path = model_dir / name
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        ready = ready and exists and size > 100
        files.append(
            {
                "name": name,
                "exists": exists,
                "size_bytes": size,
                "sha256": sha256(path) if exists and size < 32_000_000 else None,
            }
        )
    return {"model_dir": str(model_dir), "ready": ready, "files": files}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
