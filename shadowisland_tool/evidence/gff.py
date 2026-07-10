from __future__ import annotations

import re
from pathlib import Path

from ..types import GeneRecord


def parse_gff(path: Path | str | None) -> list[GeneRecord]:
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        return []

    genes: list[GeneRecord] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            seqid, _, feature_type, start, end, _, strand, _, attrs = parts
            if feature_type not in {"gene", "CDS", "tRNA", "tmRNA", "mobile_genetic_element"}:
                continue
            try:
                start_i = int(start)
                end_i = int(end)
            except ValueError:
                continue
            label = gff_label(attrs)
            genes.append(
                GeneRecord(
                    accession=seqid,
                    start=start_i,
                    end=end_i,
                    strand=strand,
                    feature_type=feature_type,
                    label=label,
                    category=classify_label(label, feature_type),
                )
            )
    return genes


def gff_label(attrs: str) -> str:
    values = []
    for key in ("gene", "Name", "product", "note", "locus_tag", "ID"):
        match = re.search(rf"(?:^|;){key}=([^;]+)", attrs)
        if match:
            values.append(match.group(1).replace("%20", " "))
    return " | ".join(values)


def classify_label(label: str, feature_type: str = "") -> str:
    text = f"{label} {feature_type}".lower()
    checks = [
        ("trna", ["trna", "transfer rna", "tmrna"]),
        ("resistance", ["resistance", "resistant", "beta-lactamase", "efflux", "multidrug"]),
        ("virulence", ["virulence", "toxin", "adhesin", "hemolysin", "fimbr", "invasion"]),
        ("phage", ["phage", "prophage", "capsid", "tail fiber", "terminase", "portal protein"]),
        ("mobility", ["integrase", "transposase", "recombinase", "insertion sequence", "conjug"]),
    ]
    for category, needles in checks:
        if any(needle in text for needle in needles):
            return category
    return "other"


def annotate_intervals(
    intervals: dict[str, list[dict[str, object]]],
    genes: list[GeneRecord],
) -> dict[str, list[dict[str, object]]]:
    genes_by_acc: dict[str, list[GeneRecord]] = {}
    for gene in genes:
        genes_by_acc.setdefault(gene.accession, []).append(gene)

    for accession, rows in intervals.items():
        acc_genes = genes_by_acc.get(accession, [])
        for row in rows:
            overlaps = [gene for gene in acc_genes if int(row["start"]) <= gene.end and gene.start <= int(row["end"])]
            counts = {name: 0 for name in ("mobility", "virulence", "resistance", "phage", "trna", "other")}
            for gene in overlaps:
                counts[gene.category] = counts.get(gene.category, 0) + 1
            evidence_classes = sum(1 for key in ("mobility", "virulence", "resistance", "phage", "trna") if counts[key])
            conf_score = evidence_confidence_score(counts, evidence_classes)
            conf_tier = "high" if conf_score >= 4 else "medium" if conf_score >= 2 else "low"
            row.update(
                {
                    "refseq_n_genes": len(overlaps),
                    "refseq_n_mobility": counts["mobility"],
                    "refseq_n_virulence": counts["virulence"],
                    "refseq_n_resistance": counts["resistance"],
                    "refseq_n_phage": counts["phage"],
                    "refseq_n_trna": counts["trna"],
                    "n_evidence_classes_y": evidence_classes,
                    "evidence_label": "predicted_GI",
                    "pred_region_rank": None,
                    "n_integrase_exts_y": counts["mobility"],
                    "n_mobility_y": counts["mobility"],
                    "evidence_rich": evidence_classes >= 2,
                    "n_resistance_markers": counts["resistance"],
                    "conf_score": conf_score,
                    "conf_tier": conf_tier,
                    "conf_tags": evidence_confidence_tags(counts, evidence_classes),
                }
            )
    return intervals


def evidence_confidence_score(counts: dict[str, int], evidence_classes: int) -> int:
    score = 0
    if counts.get("mobility", 0):
        score += 2
    if counts.get("phage", 0) or counts.get("resistance", 0) or counts.get("virulence", 0):
        score += 1
    if counts.get("trna", 0):
        score += 1
    focused = sum(counts.get(key, 0) for key in ("mobility", "virulence", "resistance", "phage", "trna"))
    if focused >= 5:
        score += 2
    elif focused >= 2:
        score += 1
    if evidence_classes >= 2:
        score += 1
    return int(score)


def evidence_confidence_tags(counts: dict[str, int], evidence_classes: int) -> str:
    tags = []
    for key in ("mobility", "phage", "resistance", "virulence", "trna"):
        if counts.get(key, 0):
            tags.append(key)
    if evidence_classes >= 2:
        tags.append("multi-evidence")
    return "|".join(tags)
