# Usage

## Command Line

```bash
shadowisland predict genome.fasta --gff annotation.gff3 --out results/
```

`--gff` is optional. When supplied, the viewer includes gene evidence tracks and
the interval tables include biological evidence summaries.

## Web UI

```bash
shadowisland serve --host 127.0.0.1 --port 8765
```

Open <http://127.0.0.1:8765> and upload a FASTA file plus an optional GFF3 file.

## Docker

```bash
docker compose up --build
```

Then open <http://127.0.0.1:8765>.

