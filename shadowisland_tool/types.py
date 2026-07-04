from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FastaRecord:
    accession: str
    description: str
    sequence: str


@dataclass
class WindowScore:
    accession: str
    start: int
    end: int
    gc: float
    p_seq: float
    p_comp: float
    p_gi: float
    logit_seq: float = 0.0
    logit_svm: float = 0.0
    y_pred: Optional[int] = None


@dataclass
class GeneRecord:
    accession: str
    start: int
    end: int
    strand: str
    feature_type: str
    label: str
    category: str

