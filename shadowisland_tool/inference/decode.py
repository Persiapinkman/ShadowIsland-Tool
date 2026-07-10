from __future__ import annotations

import statistics
from typing import Iterable

from ..types import WindowScore


STRIDE_BP = 2500
MAX_REGION_BP = 200_000


def decode_intervals(windows: Iterable[WindowScore], max_region_bp: int = MAX_REGION_BP) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[WindowScore]] = {}
    for row in windows:
        grouped.setdefault(row.accession, []).append(row)

    decoded: dict[str, list[dict[str, object]]] = {}
    for accession, rows in grouped.items():
        rows = sorted(rows, key=lambda row: row.start)
        if not rows:
            decoded[accession] = []
            continue

        has_released_labels = any(row.y_pred is not None for row in rows)
        if has_released_labels:
            selected = [row for row in rows if int(row.y_pred or 0) == 1]
            decoder = "released_workflow"
        else:
            scores = [row.p_gi for row in rows]
            tau = max(0.54, statistics.quantiles(scores, n=5)[-1] if len(scores) >= 5 else max(scores))
            selected = [row for row in rows if row.p_gi >= tau]
            decoder = "local_probability_threshold"

        intervals = []
        recommended_regions = split_long_regions(merge_windows(selected, STRIDE_BP), max_region_bp)
        for idx, (start, end, score) in enumerate(recommended_regions):
            tier = "high" if score >= 0.74 else "medium" if score >= 0.62 else "low"
            intervals.append(make_interval(accession, idx + 1, tier, start, end, score, decoder))

        if has_released_labels:
            low_regions = exploratory_regions(rows, recommended_regions, max_region_bp)
            next_idx = len(intervals) + 1
            for offset, (start, end, score) in enumerate(low_regions):
                intervals.append(make_interval(accession, next_idx + offset, "low", start, end, score, "released_workflow_exploratory"))

        decoded[accession] = sorted(intervals, key=lambda item: (int(item["start"]), int(item["end"])))
    return decoded


def make_interval(
    accession: str,
    idx: int,
    tier: str,
    start: int,
    end: int,
    score: float,
    decoder: str,
) -> dict[str, object]:
    return {
        "id": f"{accession}_pred_{idx}",
        "kind": tier,
        "start": int(start),
        "end": int(end),
        "length": int(end - start + 1),
        "score": round(float(score), 4),
        "decoder": decoder,
    }


def exploratory_regions(
    rows: list[WindowScore],
    recommended_regions: list[tuple[int, int, float]],
    max_region_bp: int,
) -> list[tuple[int, int, float]]:
    unselected = [row for row in rows if int(row.y_pred or 0) != 1]
    if not unselected:
        return []
    scores = [row.p_gi for row in unselected]
    tau = max(0.35, statistics.quantiles(scores, n=20)[-1] if len(scores) >= 20 else max(scores))
    candidates = [row for row in unselected if row.p_gi >= tau]
    regions = split_long_regions(merge_windows(candidates, STRIDE_BP), max_region_bp)
    regions = [region for region in regions if not overlaps_recommended(region, recommended_regions)]
    return sorted(regions, key=lambda region: region[2], reverse=True)[:12]


def overlaps_recommended(region: tuple[int, int, float], recommended_regions: list[tuple[int, int, float]]) -> bool:
    start, end, _ = region
    length = max(1, end - start + 1)
    for rec_start, rec_end, _ in recommended_regions:
        overlap = max(0, min(end, rec_end) - max(start, rec_start) + 1)
        if overlap / length >= 0.25:
            return True
    return False


def merge_windows(rows: list[WindowScore], max_gap: int) -> list[tuple[int, int, float]]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda row: row.start)
    merged: list[tuple[int, int, list[float]]] = []
    for row in rows:
        if not merged or row.start - merged[-1][1] > max_gap:
            merged.append((row.start, row.end, [row.p_gi]))
        else:
            start, end, scores = merged[-1]
            scores.append(row.p_gi)
            merged[-1] = (start, max(end, row.end), scores)
    return [(start, end, max(scores)) for start, end, scores in merged]


def split_long_regions(regions: list[tuple[int, int, float]], max_len: int) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    for start, end, score in regions:
        cursor = start
        while end - cursor + 1 > max_len:
            out.append((cursor, cursor + max_len - 1, score))
            cursor += max_len
        out.append((cursor, end, score))
    return out
