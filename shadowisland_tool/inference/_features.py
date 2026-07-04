from __future__ import annotations

import itertools
import sys

import numpy as np


WORD_SIZE = 4
FRAGMENT_LENGTH = 6000
TOKEN_LEN = FRAGMENT_LENGTH - WORD_SIZE + 1
STRIDE_BP = 2500
BASES = "ACGT"
KMER4_TO_ID = {"".join(p): i + 1 for i, p in enumerate(itertools.product(BASES, repeat=WORD_SIZE))}


def numpy_pickle_compat() -> None:
    """Allow loading sklearn artifacts pickled with numpy 2 on numpy 1.x."""
    try:
        import numpy._core  # type: ignore  # noqa: F401
        return
    except Exception:
        pass
    sys.modules.setdefault("numpy._core", np.core)
    for name in ("multiarray", "numeric", "umath", "fromnumeric", "_multiarray_umath"):
        try:
            sys.modules.setdefault(f"numpy._core.{name}", __import__(f"numpy.core.{name}", fromlist=["*"]))
        except Exception:
            continue


def tokenize_4mer(seq: str) -> np.ndarray:
    tokens = np.zeros(TOKEN_LEN, dtype=np.int64)
    seq = seq.upper()
    for i in range(TOKEN_LEN):
        kmer = seq[i : i + WORD_SIZE]
        tokens[i] = KMER4_TO_ID.get(kmer, 0) if "N" not in kmer else 0
    return tokens


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def kmer_profile_keys(k: int) -> list[str]:
    profile_keys = {min("".join(p), reverse_complement("".join(p))) for p in itertools.product(BASES, repeat=k)}
    return sorted(profile_keys)


def kmer_profile(seq: str, keys: list[str]) -> np.ndarray:
    key_to_idx = {key: i for i, key in enumerate(keys)}
    counts = np.zeros(len(keys), dtype=np.float32)
    total = 0
    seq = seq.upper().replace("-", "")
    for i in range(0, len(seq) - 7 + 1):
        kmer = seq[i : i + 7]
        if any(base not in BASES for base in kmer):
            continue
        counts[key_to_idx[min(kmer, reverse_complement(kmer))]] += 1.0
        total += 1
    if total:
        counts /= float(total)
    return counts


def gc_fraction(seq: str) -> float:
    bases = [base for base in seq.upper() if base in BASES]
    if not bases:
        return 0.0
    return (bases.count("G") + bases.count("C")) / len(bases)
