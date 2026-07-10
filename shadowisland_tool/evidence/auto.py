from __future__ import annotations

import re
from dataclasses import dataclass

from ..types import FastaRecord, GeneRecord


START_CODONS = {"ATG", "GTG", "TTG"}
STOP_CODONS = {"TAA", "TAG", "TGA"}
MIN_ORF_NT = 270
MAX_ORFS_PER_RECORD = 15000

GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


@dataclass(frozen=True)
class OrfCandidate:
    accession: str
    start: int
    end: int
    strand: str
    nt_sequence: str

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class FunctionalCall:
    category: str
    label: str


def auto_annotate_records(records: list[FastaRecord]) -> list[GeneRecord]:
    """Generate lightweight FASTA-only gene/function evidence.

    This is a self-contained fallback for web uploads without GFF3. It calls
    bacterial-style ORFs on both strands, then labels a conservative subset of
    ORFs with sequence-motif based functional evidence classes used by the
    viewer: mobility, resistance, phage and virulence. Calls are intentionally
    marked as putative and should be treated as screening evidence, not curated
    annotation.
    """

    genes: list[GeneRecord] = []
    for record in records:
        candidates = select_orfs(find_orfs(record))
        for idx, candidate in enumerate(candidates, start=1):
            protein = translate(candidate.nt_sequence)
            call = classify_orf(protein)
            aa_len = len(protein.rstrip("*"))
            label = f"SI_AUTO_{idx:05d} | {call.label} | {aa_len} aa | FASTA-only motif scan"
            genes.append(
                GeneRecord(
                    accession=candidate.accession,
                    start=candidate.start,
                    end=candidate.end,
                    strand=candidate.strand,
                    feature_type="auto_CDS",
                    label=label,
                    category=call.category,
                )
            )
    return genes


def find_orfs(record: FastaRecord, min_nt: int = MIN_ORF_NT) -> list[OrfCandidate]:
    seq = record.sequence.upper()
    candidates: list[OrfCandidate] = []
    for strand, scan_seq in (("+", seq), ("-", reverse_complement(seq))):
        seq_len = len(scan_seq)
        for frame in range(3):
            start_pos: int | None = None
            for pos in range(frame, seq_len - 2, 3):
                codon = scan_seq[pos : pos + 3]
                if start_pos is None:
                    if codon in START_CODONS:
                        start_pos = pos
                    continue
                if codon in STOP_CODONS:
                    if pos + 3 - start_pos >= min_nt:
                        nt = scan_seq[start_pos : pos + 3]
                        start, end = to_forward_coords(start_pos, pos + 3, seq_len, strand)
                        candidates.append(OrfCandidate(record.accession, start, end, strand, nt))
                    start_pos = None
    return candidates


def select_orfs(candidates: list[OrfCandidate]) -> list[OrfCandidate]:
    selected: list[OrfCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.length, reverse=True):
        if any(overlap_fraction(candidate, existing) > 0.72 for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) >= MAX_ORFS_PER_RECORD:
            break
    return sorted(selected, key=lambda item: (item.accession, item.start, item.end, item.strand))


def overlap_fraction(a: OrfCandidate, b: OrfCandidate) -> float:
    if a.accession != b.accession:
        return 0.0
    overlap = max(0, min(a.end, b.end) - max(a.start, b.start) + 1)
    return overlap / max(1, min(a.length, b.length))


def to_forward_coords(start0: int, stop0: int, seq_len: int, strand: str) -> tuple[int, int]:
    if strand == "+":
        return start0 + 1, stop0
    return seq_len - stop0 + 1, seq_len - start0


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1].upper()


def translate(nt: str) -> str:
    return "".join(GENETIC_CODE.get(nt[i : i + 3], "X") for i in range(0, len(nt) - 2, 3))


def classify_orf(protein: str) -> FunctionalCall:
    protein = protein.rstrip("*")
    aa_len = len(protein)

    if is_integrase_like(protein, aa_len):
        return FunctionalCall("mobility", "putative integrase/recombinase mobility marker")
    if is_beta_lactamase_like(protein, aa_len):
        return FunctionalCall("resistance", "putative beta-lactamase resistance marker")
    if is_efflux_like(protein, aa_len):
        return FunctionalCall("resistance", "putative multidrug efflux resistance marker")
    if is_surface_or_rtx_like(protein, aa_len):
        return FunctionalCall("virulence", "putative adhesin/toxin-associated virulence marker")
    return FunctionalCall("other", "predicted ORF")


def is_integrase_like(protein: str, aa_len: int) -> bool:
    if not 220 <= aa_len <= 520:
        return False
    return bool(re.search(r"R.{2}H.{2}R.{80,180}Y", protein))


def is_transposase_like(protein: str, aa_len: int) -> bool:
    if not 180 <= aa_len <= 1000:
        return False
    dde = re.search(r"D.{35,140}D.{18,140}E", protein)
    helix_turn_helix = re.search(r"[KR].{2,8}[ST].{2,8}[KR].{6,24}[WFY]", protein[:260])
    return bool(dde and helix_turn_helix)


def is_beta_lactamase_like(protein: str, aa_len: int) -> bool:
    if not 230 <= aa_len <= 420:
        return False
    return bool(re.search(r"S..K", protein) and re.search(r"S[DT]N", protein) and re.search(r"KT[GS]", protein))


def is_efflux_like(protein: str, aa_len: int) -> bool:
    if not 320 <= aa_len <= 650:
        return False
    return count_hydrophobic_segments(protein) >= 8 and "G" in protein[80:220]


def is_phage_terminase_like(protein: str, aa_len: int) -> bool:
    if not 320 <= aa_len <= 760:
        return False
    walker_a = re.search(r"[AG].{3,5}G[KT]T", protein)
    nuclease = re.search(r"D.{20,90}E.{15,90}[DE]", protein)
    return bool(walker_a and nuclease)


def is_surface_or_rtx_like(protein: str, aa_len: int) -> bool:
    if aa_len >= 350 and re.search(r"LP.TG.{0,35}$", protein) and has_signal_like_n_terminus(protein):
        return True
    return len(re.findall(r"GG.G.D", protein)) >= 3


def count_hydrophobic_segments(protein: str, min_len: int = 17) -> int:
    return len(re.findall(rf"[AILMFWVY]{{{min_len},}}", protein))


def has_signal_like_n_terminus(protein: str) -> bool:
    return bool(re.search(r"^[MKR]{1,8}.{3,25}[AILMFWV]{8,}", protein[:70]))
