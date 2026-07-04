#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shadowisland_tool.pipeline import DEFAULT_MODEL_DIR  # noqa: E402
from shadowisland_tool.scripts_support import check_model_bundle  # noqa: E402


def main() -> int:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL_DIR
    payload = check_model_bundle(model_dir)
    print(json.dumps(payload, indent=2))
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

