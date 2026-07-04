from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .evidence.gff import annotate_intervals, parse_gff
from .inference.decode import decode_intervals
from .inference.released_workflow import predict_window_scores
from .io import parse_fasta
from .report.writers import write_result_package
from .types import FastaRecord, WindowScore


TOOL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = TOOL_ROOT / "model_bundle" / "paper_model_v1"


@dataclass
class PredictionResult:
    out_dir: Path
    records: list[FastaRecord]
    n_windows: int
    n_intervals: int


class ShadowIslandPredictor:
    """Saved-weight ShadowIsland predictor.

    This is the stable API used by the CLI and web service.
    """

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path = "paper-model-v1",
    ) -> "ShadowIslandPredictor":
        if str(model) in {"paper-model-v1", "default"}:
            return cls(DEFAULT_MODEL_DIR)
        return cls(Path(model))

    def predict_fasta(
        self,
        fasta: Path | str,
        *,
        gff: Path | str | None = None,
        out_dir: Path | str,
    ) -> PredictionResult:
        records = parse_fasta(fasta)
        if not records:
            raise ValueError(f"No FASTA records found: {fasta}")
        return self.predict_records(records, gff=gff, out_dir=out_dir, input_fasta=Path(fasta))

    def predict_records(
        self,
        records: list[FastaRecord],
        *,
        gff: Path | str | None = None,
        out_dir: Path | str,
        input_fasta: Path | None = None,
    ) -> PredictionResult:
        self._check_model_bundle()
        raw_rows = predict_window_scores(records, self.model_dir)
        windows = [WindowScore(**row) for row in raw_rows]
        intervals = decode_intervals(windows)

        genes = parse_gff(gff)
        annotate_intervals(intervals, genes)

        out_path = Path(out_dir)
        provenance = {
            "runner": "released_manuscript_workflow",
            "model_bundle": str(self.model_dir),
            "model_bundle_ready": True,
            "model_bundle_version": "paper_model_v1",
            "workflow": "released manuscript prediction workflow",
            "input_fasta": str(input_fasta) if input_fasta else None,
            "input_gff": str(gff) if gff else None,
            "confidence_tier_note": (
                "Uploaded genomes assign high/medium/low from available GFF/RefSeq-style evidence."
            ),
        }
        write_result_package(out_path, records=records, windows=windows, intervals=intervals, genes=genes, provenance=provenance)
        return PredictionResult(
            out_dir=out_path,
            records=records,
            n_windows=len(windows),
            n_intervals=sum(len(rows) for rows in intervals.values()),
        )

    def _check_model_bundle(self) -> None:
        required = [
            "paper_model_v1_encoder.pth",
            "paper_model_v1_classifier.pth",
            "paper_model_v1_decoder.pt",
            "paper_model_v1_feature_scaler.pkl",
            "paper_model_v1_feature_model.pkl",
            "paper_model_v1_calibration.bin",
        ]
        missing = [name for name in required if not (self.model_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model files under {self.model_dir}: {', '.join(missing)}. "
                "Download the release model bundle or provide --model-dir."
            )
