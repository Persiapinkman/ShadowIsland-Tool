from __future__ import annotations

import json
from importlib import resources

from ..types import FastaRecord, GeneRecord

CATEGORY_BY_CODE = {
    0: "other",
    1: "mobility",
    2: "virulence",
    3: "resistance",
    4: "phage",
    5: "trna",
}

LABEL_BY_CATEGORY = {
    "other": "packaged RefSeq CDS",
    "mobility": "packaged RefSeq mobility marker",
    "virulence": "packaged RefSeq virulence marker",
    "resistance": "packaged RefSeq resistance marker",
    "phage": "packaged RefSeq phage marker",
    "trna": "packaged RefSeq tRNA marker",
}


def packaged_reference_genes(records: list[FastaRecord]) -> list[GeneRecord]:
    """Return packaged reference gene evidence for known demo accessions."""

    payload = load_packaged_refseq_payload()
    genes: list[GeneRecord] = []
    for record in records:
        entry = payload.get(record.accession)
        if not entry:
            continue
        for idx, row in enumerate(entry.get("genes", []), start=1):
            if len(row) < 4:
                continue
            start, end, strand_code, category_code = row[:4]
            category = CATEGORY_BY_CODE.get(int(category_code), "other")
            genes.append(
                GeneRecord(
                    accession=record.accession,
                    start=int(start),
                    end=int(end),
                    strand="+" if int(strand_code) == 1 else "-",
                    feature_type="reference_CDS",
                    label=f"REFSEQ_{idx:05d} | {LABEL_BY_CATEGORY[category]}",
                    category=category,
                )
            )
    return genes


def load_packaged_refseq_payload() -> dict[str, dict[str, object]]:
    try:
        text = (
            resources.files("shadowisland_tool.assets.web")
            .joinpath("canonical_case_circos/refseq_genes.js")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError):
        return {}
    prefix = "window.REFSEQ_GENES = "
    text = text.strip()
    if text.startswith(prefix):
        text = text[len(prefix) :]
    if text.endswith(";"):
        text = text[:-1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
