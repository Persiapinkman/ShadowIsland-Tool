#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python scripts/check_model_bundle.py
python -m shadowisland_tool.cli predict examples/sample.fasta \
  --gff examples/sample.gff3 \
  --out runs/smoke

test -f runs/smoke/window_probs.csv
test -f runs/smoke/predicted_intervals.csv
test -f runs/smoke/viewer/index.html
echo "Smoke test complete: runs/smoke/viewer/index.html"
