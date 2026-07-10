# ShadowIsland Tool

Standalone ShadowIsland inference and evidence-visualization tool.

The public release provides the manuscript-version prediction workflow for:

```text
FASTA input -> genomic island prediction -> GFF3 evidence annotation when provided
            -> FASTA-only ORF/motif evidence scan when GFF3 is omitted
            -> tables, report and circular evidence viewer
```

This repository is intended for running the released tool. Training scripts,
research development files and manuscript working files are not included.

## Quick Start

```bash
cd ShadowIsland-Tool
python -m venv .venv
source .venv/bin/activate
pip install -e ".[web]"
python scripts/check_model_bundle.py

shadowisland predict examples/sample.fasta \
  --gff examples/sample.gff3 \
  --out runs/sample
```

Open:

```text
runs/sample/viewer/index.html
```

## Web Service

```bash
shadowisland serve --host 127.0.0.1 --port 8765
```

Then open <http://127.0.0.1:8765>.

If only FASTA is uploaded, the service now automatically runs a built-in bacterial ORF caller plus conservative motif scan to generate putative mobility, phage, resistance and virulence evidence for the circular viewer. Uploading GFF3 is still preferred when curated annotation is available.

## Docker

```bash
docker compose up --build
```

The image runs CPU inference by default. GPU builds can use the same package
with a CUDA-enabled PyTorch installation.

## Outputs

Each prediction run writes:

```text
out/
  window_probs.csv
  predicted_intervals.csv
  gene_catalog.csv
  region_summary.csv
  functional_report.md
  provenance.json
  viewer/
    index.html
    case_data.js
    refseq_genes.js
    gc_windows.js
```

## Model Bundle

Large model files are distributed separately from the git repository. Download
the manuscript-version model bundle from the release assets and unpack it into:

```text
model_bundle/paper_model_v1/
```

Then run:

```bash
python scripts/check_model_bundle.py
```
