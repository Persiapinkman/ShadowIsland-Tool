from __future__ import annotations

import re
from pathlib import Path

from .types import FastaRecord


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "upload.dat"


def parse_fasta(path: Path | str) -> list[FastaRecord]:
    path = Path(path)
    records: list[FastaRecord] = []
    header: str | None = None
    chunks: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append(_record_from_parts(header, chunks))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        records.append(_record_from_parts(header, chunks))
    return records


def _record_from_parts(header: str, chunks: list[str]) -> FastaRecord:
    accession = header.split()[0].split("|")[0]
    seq = re.sub(r"[^ACGTNacgtn]", "N", "".join(chunks)).upper()
    return FastaRecord(accession=accession, description=header, sequence=seq)


def gc_fraction(seq: str) -> float:
    bases = [base for base in seq.upper() if base in "ACGT"]
    if not bases:
        return 0.0
    return (bases.count("G") + bases.count("C")) / len(bases)

