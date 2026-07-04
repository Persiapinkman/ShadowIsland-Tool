"""Patch index generators for IGLOO v2."""

from __future__ import annotations

import numpy as np


def generate_patches(
    patch_mode: str,
    patch_size: int,
    nb_patches: int,
    vector_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Return patch index array [nb_patches, patch_size]."""
    mode = patch_mode.lower()
    if mode == "random":
        return gen_random_patches(patch_size, nb_patches, vector_size, seed)

    if mode == "grid":
        return gen_grid_patches(patch_size, nb_patches, vector_size)

    if mode == "hybrid":
        return gen_hybrid_patches(patch_size, nb_patches, vector_size, seed)

    raise ValueError(
        f"Unknown patch_mode={patch_mode!r}. Use 'random', 'grid', or 'hybrid'."
    )


def gen_random_patches(
    patch_size: int,
    nb_patches: int,
    vector_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Seeded copy of the original IGLOO random patch generator."""
    rng = np.random.RandomState(seed)
    step = int(vector_size) - 1
    collect = [
        sorted(rng.choice(range(step + 1), patch_size, replace=False).tolist())
        for _ in range(nb_patches)
    ]
    return np.asarray(collect, dtype=np.int64)


def gen_grid_patches(
    patch_size: int,
    nb_patches: int,
    vector_size: int,
) -> np.ndarray:
    """
    Grid patches: stride-3 sliding window for ~full coverage (2100 > 5997/3),
    then multi-scale fillers for the remaining budget.
    """
    max_start = max(vector_size - patch_size, 0)
    collect: list[list[int]] = []

    stride = 3  # 4-mer patches every 3 positions → covers 5997 in ~2000 patches
    for start in range(0, max_start + 1, stride):
        collect.append([start + k for k in range(patch_size)])
        if len(collect) >= nb_patches:
            return np.array(collect[:nb_patches], dtype=np.int64)

    # Fill any remaining budget with multi-scale patches
    scales = [1, 16, 64]
    si = 0
    while len(collect) < nb_patches:
        scale = scales[si % len(scales)]
        span = (patch_size - 1) * scale
        ms = max(vector_size - 1 - span, 0)
        n_rem = nb_patches - len(collect)
        j = si // len(scales)
        start = int(round(j * ms / max(n_rem - 1, 1)))
        patch = [min(start + k * scale, vector_size - 1) for k in range(patch_size)]
        collect.append(patch)
        si += 1

    return np.array(collect[:nb_patches], dtype=np.int64)


def gen_hybrid_patches(
    patch_size: int,
    nb_patches: int,
    vector_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Hybrid patches: regular coverage first, seeded random diversity second."""
    coverage_budget = (vector_size + max(patch_size - 1, 1) - 1) // max(patch_size - 1, 1)
    n_grid = min(nb_patches, coverage_budget)
    n_random = nb_patches - n_grid
    grid = gen_grid_patches(patch_size, n_grid, vector_size)

    if n_random == 0:
        return grid

    rng = np.random.RandomState(seed)
    random_rows = [
        sorted(rng.choice(vector_size, patch_size, replace=False).tolist())
        for _ in range(n_random)
    ]
    random = np.array(random_rows, dtype=np.int64)
    return np.concatenate([grid, random], axis=0)


def patch_coverage_stats(patches: np.ndarray, vector_size: int) -> dict:
    coverage = np.zeros(vector_size, dtype=np.int32)
    for row in patches:
        for idx in row:
            coverage[int(idx)] += 1
    return {
        "never_sampled": int((coverage == 0).sum()),
        "never_sampled_frac": float((coverage == 0).mean()),
        "max_coverage": int(coverage.max()),
    }
