from __future__ import annotations

import pickle
import site
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.append(user_site)
    import torch
    import torch.nn as nn

from ._features import (
    FRAGMENT_LENGTH,
    STRIDE_BP,
    TOKEN_LEN,
    gc_fraction,
    kmer_profile,
    kmer_profile_keys,
    numpy_pickle_compat,
    tokenize_4mer,
)


MODEL_LIB = Path(__file__).resolve().parent / "r2_lib"
if str(MODEL_LIB) not in sys.path:
    sys.path.insert(0, str(MODEL_LIB))

from igloo_v2.config import get_model_config  # noqa: E402
from models.encoder_v2 import EncoderV2  # noqa: E402


PROFILE_A = {"tau": 0.25, "median_threshold": 0.30, "top_frac": 0.20}
PROFILE_B = {"tau": 0.40, "median_threshold": 0.60, "top_frac": 0.02}
MAX_LEN_BP = 200_000
FEATURE_CODE_MAP = {
    1: "n_windows",
    2: "genome_len",
    3: "seq_q50",
    4: "svm_q50",
    5: "seq_q95",
    6: "svm_q95",
    7: "seq_frac_ge_090",
    8: "svm_frac_ge_090",
    9: "seq_minus_svm_q50",
    10: "seq_minus_svm_q95",
    11: "svm_minus_seq_q95",
}


class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int = 512, dropout: float = 0.2, num_classes: int = 2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class PLCRF(nn.Module):
    def __init__(self, feat_dim: int = 2, num_states: int = 2):
        super().__init__()
        self.K = num_states
        self.emitter = nn.Linear(feat_dim, num_states)
        self.trans = nn.Parameter(torch.zeros(num_states, num_states))
        self.start = nn.Parameter(torch.zeros(num_states))
        self.end = nn.Parameter(torch.zeros(num_states))

    def emissions(self, feats: torch.Tensor) -> torch.Tensor:
        return self.emitter(feats)

    @torch.no_grad()
    def forward_backward_marginals(self, feats: torch.Tensor) -> torch.Tensor:
        emissions = self.emissions(feats)
        t_len, k = emissions.shape
        log_alpha = torch.empty((t_len, k), device=feats.device, dtype=emissions.dtype)
        log_beta = torch.empty((t_len, k), device=feats.device, dtype=emissions.dtype)
        log_alpha[0] = self.start + emissions[0]
        for t in range(1, t_len):
            log_alpha[t] = torch.logsumexp(log_alpha[t - 1].view(k, 1) + self.trans, dim=0) + emissions[t]
        log_beta[t_len - 1] = self.end
        for t in range(t_len - 2, -1, -1):
            log_beta[t] = torch.logsumexp(self.trans + emissions[t + 1].view(1, k) + log_beta[t + 1].view(1, k), dim=1)
        log_z = torch.logsumexp(log_alpha[t_len - 1] + self.end, dim=0)
        log_marg = log_alpha + log_beta - log_z
        return torch.softmax(log_marg, dim=1)


@dataclass(frozen=True)
class WindowFrame:
    accession: str
    starts0: np.ndarray
    ends0: np.ndarray
    p_gi: np.ndarray
    p_seq: np.ndarray
    p_svm: np.ndarray
    logit_seq: np.ndarray
    logit_svm: np.ndarray
    selected_mask: np.ndarray


@dataclass(frozen=True)
class CalibrationRules:
    feature_codes: np.ndarray
    op_codes: np.ndarray
    thresholds: np.ndarray
    blend_weights: np.ndarray


class ReleasedWorkflow:
    def __init__(self, bundle_dir: Path, device: str = "auto"):
        self.bundle_dir = Path(bundle_dir)
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg = get_model_config("paper")
        self.encoder = EncoderV2(cfg).to(self.device)
        self.head = ClassifierHead(dropout=0.2).to(self.device)
        self.crf = PLCRF(2, 2).to(self.device)

        enc_state = torch.load(self.bundle_dir / "paper_model_v1_encoder.pth", map_location=self.device)
        self.encoder.load_state_dict({k.replace("module.", ""): v for k, v in enc_state.items()}, strict=True)
        head_state = torch.load(self.bundle_dir / "paper_model_v1_classifier.pth", map_location=self.device)
        self.head.load_state_dict({k.replace("module.", ""): v for k, v in head_state.items()}, strict=True)
        crf_obj = torch.load(self.bundle_dir / "paper_model_v1_decoder.pt", map_location=self.device)
        self.crf.load_state_dict(crf_obj.get("state_dict", crf_obj), strict=True)
        self.encoder.eval()
        self.head.eval()
        self.crf.eval()

        numpy_pickle_compat()
        with (self.bundle_dir / "paper_model_v1_feature_scaler.pkl").open("rb") as handle:
            self.scaler = pickle.load(handle)
        with (self.bundle_dir / "paper_model_v1_feature_model.pkl").open("rb") as handle:
            svm_obj = pickle.load(handle)
        self.svm = svm_obj["model"] if isinstance(svm_obj, dict) and "model" in svm_obj else svm_obj
        self.profile_keys = kmer_profile_keys(7)
        self.rules = load_calibration(self.bundle_dir / "paper_model_v1_calibration.bin")

    @torch.no_grad()
    def predict_records(self, records: list[object], batch_size: int = 64) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for record in records:
            frame = self.predict_record(record, batch_size=batch_size)
            for i in range(len(frame.starts0)):
                start0 = int(frame.starts0[i])
                end0 = int(frame.ends0[i])
                rows.append(
                    {
                        "accession": frame.accession,
                        "start": start0 + 1,
                        "end": end0,
                        "gc": gc_fraction(str(record.sequence)[start0:end0]),
                        "p_seq": float(frame.p_seq[i]),
                        "p_comp": float(frame.p_svm[i]),
                        "p_gi": float(frame.p_gi[i]),
                        "logit_seq": float(frame.logit_seq[i]),
                        "logit_svm": float(frame.logit_svm[i]),
                        "y_pred": int(frame.selected_mask[i]),
                    }
                )
        return rows

    def predict_record(self, record: object, batch_size: int) -> WindowFrame:
        accession = str(record.accession)
        sequence = str(record.sequence).upper()
        starts0 = np.asarray(list(range(0, max(len(sequence) - FRAGMENT_LENGTH + 1, 1), STRIDE_BP)), dtype=np.int64)
        ends0 = np.asarray([min(int(s) + FRAGMENT_LENGTH, len(sequence)) for s in starts0], dtype=np.int64)
        token_buf = np.empty((len(starts0), TOKEN_LEN), dtype=np.int64)
        seq_buf: list[str] = []
        for i, start0 in enumerate(starts0):
            frag = sequence[int(start0) : int(start0) + FRAGMENT_LENGTH]
            if len(frag) < FRAGMENT_LENGTH:
                frag = frag + "N" * (FRAGMENT_LENGTH - len(frag))
            seq_buf.append(frag)
            token_buf[i] = tokenize_4mer(frag)

        logit_seq, p_seq = self.predict_sequence_branch(token_buf, batch_size)
        logit_svm, p_svm = self.predict_svm_branch(seq_buf)
        blend = choose_blend_weight(accession, p_seq, p_svm, len(starts0), len(sequence), self.rules)
        feats = np.stack([logit_seq * (2.0 * blend), logit_svm * (2.0 * (1.0 - blend))], axis=1).astype(np.float32)
        with torch.no_grad():
            marg = self.crf.forward_backward_marginals(torch.from_numpy(feats).to(self.device))[:, 1]
        p_gi = marg.detach().cpu().numpy().astype(np.float32)
        selected_mask = released_interval_mask(starts0, ends0, p_gi, p_seq, p_svm, blend)
        return WindowFrame(accession, starts0, ends0, p_gi, p_seq, p_svm, logit_seq, logit_svm, selected_mask)

    def predict_sequence_branch(self, token_buf: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        logits_all = np.empty(token_buf.shape[0], dtype=np.float32)
        probs_all = np.empty(token_buf.shape[0], dtype=np.float32)
        for start in range(0, token_buf.shape[0], batch_size):
            batch = torch.from_numpy(token_buf[start : start + batch_size]).to(self.device)
            logits = self.head(self.encoder(batch))
            probs = torch.softmax(logits, dim=1)[:, 1]
            logits_all[start : start + batch_size] = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
            probs_all[start : start + batch_size] = probs.detach().cpu().numpy()
        return logits_all, probs_all

    def predict_svm_branch(self, seqs: list[str]) -> tuple[np.ndarray, np.ndarray]:
        x = np.vstack([kmer_profile(seq, self.profile_keys) for seq in seqs])
        x_scaled = self.scaler.transform(x)
        if hasattr(self.svm, "predict_proba"):
            p = self.svm.predict_proba(x_scaled)[:, 1].astype(np.float32)
        else:
            scores = self.svm.decision_function(x_scaled)
            p = (1.0 / (1.0 + np.exp(-scores))).astype(np.float32)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).astype(np.float32), p


def load_calibration(path: Path) -> CalibrationRules:
    with path.open("rb") as handle:
        payload = np.load(handle)
        return CalibrationRules(
            feature_codes=payload["feature_codes"].astype(np.int16),
            op_codes=payload["op_codes"].astype(np.int8),
            thresholds=payload["thresholds"].astype(np.float32),
            blend_weights=payload["blend_weights"].astype(np.float32),
        )


def choose_blend_weight(accession: str, p_seq: np.ndarray, p_svm: np.ndarray, n_windows: int, genome_len: int, rules: CalibrationRules) -> float:
    diff = p_seq - p_svm
    features = {
        "accession": accession,
        "n_windows": float(n_windows),
        "genome_len": float(genome_len),
        "seq_q50": float(np.quantile(p_seq, 0.50)),
        "svm_q50": float(np.quantile(p_svm, 0.50)),
        "seq_q95": float(np.quantile(p_seq, 0.95)),
        "svm_q95": float(np.quantile(p_svm, 0.95)),
        "seq_frac_ge_090": float((p_seq >= 0.90).mean()),
        "svm_frac_ge_090": float((p_svm >= 0.90).mean()),
        "seq_minus_svm_q50": float(np.quantile(diff, 0.50)),
        "seq_minus_svm_q95": float(np.quantile(diff, 0.95)),
        "svm_minus_seq_q95": float(np.quantile(-diff, 0.95)),
    }
    blend = 0.5
    for feature_code, op_code, threshold, blend_weight in zip(
        rules.feature_codes, rules.op_codes, rules.thresholds, rules.blend_weights
    ):
        value = float(features[FEATURE_CODE_MAP[int(feature_code)]])
        cond = value <= float(threshold) if int(op_code) == 0 else value >= float(threshold)
        if cond:
            blend = float(blend_weight)
    return blend


def released_interval_mask(starts0: np.ndarray, ends0: np.ndarray, p_gi: np.ndarray, p_seq: np.ndarray, p_svm: np.ndarray, blend: float) -> np.ndarray:
    frame = [{"window_idx": i, "start": int(starts0[i]), "end": int(ends0[i]), "p_gi": float(p_gi[i])} for i in range(len(starts0))]
    if blend <= 0.75:
        selected = [row for row in frame if row["p_gi"] >= 0.5]
        regions, _ = _bounded_segments(selected)
    else:
        stats = score_stats(p_gi)
        params = PROFILE_A if _choose_profile(stats) == "profile_a" else PROFILE_B
        selected = _candidate_windows(frame, p_gi, params)
        regions, _ = _bounded_segments(selected)
        if stats["q50"] >= 0.95:
            branch = [
                {**frame[i], "p_final": float(p_gi[i]), "p_min_branch": float(min(p_seq[i], p_svm[i]))}
                for i in range(len(frame))
                if p_gi[i] >= 0.98 and min(p_seq[i], p_svm[i]) >= 0.90
            ]
            extra, _ = _secondary_segments(branch)
            regions = merge_halfopen(regions + extra)
    mask = np.zeros(len(starts0), dtype=np.int64)
    for start, end in regions:
        inside = np.where((starts0 >= start) & (ends0 <= end))[0]
        mask[inside] = 1
    return mask


def score_stats(p: np.ndarray) -> dict[str, float]:
    return {
        "q25": float(np.quantile(p, 0.25)),
        "q50": float(np.quantile(p, 0.50)),
        "q75": float(np.quantile(p, 0.75)),
        "q95": float(np.quantile(p, 0.95)),
        "q99": float(np.quantile(p, 0.99)),
        "frac_ge_098": float((p >= 0.98).mean()),
    }


def _choose_profile(stats: dict[str, float]) -> str:
    q50 = stats["q50"]
    q95 = stats["q95"]
    frac98 = stats["frac_ge_098"]
    if q50 >= 0.95:
        return "profile_b"
    if 0.75 <= q50 < 0.95:
        return "profile_a"
    if q50 <= 0.06 and q95 >= 0.95 and frac98 >= 0.08:
        return "profile_a"
    if 0.10 <= q50 <= 0.17 and q95 >= 0.89:
        return "profile_a"
    return "profile_b"


def _candidate_windows(frame: list[dict[str, object]], p_gi: np.ndarray, params: dict[str, float]) -> list[dict[str, object]]:
    if float(np.median(p_gi)) >= float(params["median_threshold"]):
        n = max(1, int(len(frame) * float(params["top_frac"])))
        return sorted(frame, key=lambda row: float(row["p_gi"]), reverse=True)[:n]
    return [row for row in frame if float(row["p_gi"]) >= float(params["tau"])]


def selected_to_regions_basic(selected: list[dict[str, object]]) -> list[tuple[int, int]]:
    if not selected:
        return []
    regions: list[list[int]] = []
    current: list[int] | None = None
    for row in sorted(selected, key=lambda item: int(item["start"])):
        start, end = int(row["start"]), int(row["end"])
        if current is None or start > current[1]:
            if current is not None:
                regions.append(current)
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    if current is not None:
        regions.append(current)
    return [(start, end) for start, end in regions]


def _bounded_segments(selected: list[dict[str, object]], max_len_bp: int = MAX_LEN_BP) -> tuple[list[tuple[int, int]], int]:
    working = list(selected)
    removed = 0
    while True:
        regions = selected_to_regions_basic(working)
        too_long = [(start, end) for start, end in regions if end - start > max_len_bp]
        if not too_long:
            return regions, removed
        start, end = max(too_long, key=lambda item: item[1] - item[0])
        inside = [row for row in working if int(row["start"]) >= start and int(row["end"]) <= end]
        if len(inside) <= 1:
            return regions, removed
        midpoint = (start + end) / 2
        drop = sorted(
            inside,
            key=lambda row: (float(row["p_gi"]), abs(((int(row["start"]) + int(row["end"])) / 2) - midpoint), int(row["window_idx"])),
        )[0]
        working.remove(drop)
        removed += 1


def _secondary_segments(selected: list[dict[str, object]]) -> tuple[list[tuple[int, int]], int]:
    if not selected:
        return [], 0
    prepared = [{**row, "p_gi": float(row.get("p_final", row["p_gi"]))} for row in selected]
    regions, removed = _bounded_segments(prepared)
    regions = [region for region in regions if region[1] - region[0] >= 4000]
    regions = sorted(regions, key=lambda region: score_region(region, prepared), reverse=True)[:3]
    return merge_halfopen(regions), removed


def score_region(region: tuple[int, int], selected: list[dict[str, object]]) -> tuple[float, int, int]:
    start, end = region
    sub = [row for row in selected if int(row["start"]) >= start and int(row["end"]) <= end]
    score = max((float(row.get("p_final", row["p_gi"])) for row in sub), default=0.0)
    return score, len(sub), end - start


def merge_halfopen(intervals: list[tuple[int, int]], gap_bp: int = 0) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1] + gap_bp:
            merged.append([int(start), int(end)])
        else:
            merged[-1][1] = max(merged[-1][1], int(end))
    return [(start, end) for start, end in merged]
