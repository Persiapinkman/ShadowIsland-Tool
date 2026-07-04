from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import DEFAULT_MODEL_DIR, ShadowIslandPredictor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shadowisland")
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="Run saved-weight ShadowIsland inference.")
    predict.add_argument("fasta", type=Path)
    predict.add_argument("--gff", type=Path, default=None)
    predict.add_argument("--out", type=Path, required=True)
    predict.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)

    serve = sub.add_parser("serve", help="Run the optional local web service.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--reload", action="store_true")

    check = sub.add_parser("check-model", help="Validate model bundle files.")
    check.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)

    args = parser.parse_args(argv)
    if args.command == "predict":
        predictor = ShadowIslandPredictor(args.model_dir)
        result = predictor.predict_fasta(args.fasta, gff=args.gff, out_dir=args.out)
        print(
            json.dumps(
                {
                    "out_dir": str(result.out_dir),
                    "n_records": len(result.records),
                    "n_windows": result.n_windows,
                    "n_intervals": result.n_intervals,
                    "viewer": str(result.out_dir / "viewer" / "index.html"),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "serve":
        try:
            import uvicorn
        except ModuleNotFoundError as exc:
            raise SystemExit("Install web extras first: pip install -e '.[web]'") from exc
        uvicorn.run("shadowisland_tool.web.server:app", host=args.host, port=args.port, reload=args.reload)
        return 0
    if args.command == "check-model":
        from .scripts_support import check_model_bundle

        payload = check_model_bundle(args.model_dir)
        print(json.dumps(payload, indent=2))
        return 0 if payload["ready"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
