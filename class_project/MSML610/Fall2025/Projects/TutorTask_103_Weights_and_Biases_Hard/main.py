"""
Project CLI entrypoint.

Examples (inside Docker):
  python main.py train
  python main.py tune --model xgboost
  python main.py serve --host 0.0.0.0 --port 8000

Notes:
- W&B mode is controlled via env vars, e.g. WANDB_MODE=offline|online
- This CLI does not rebuild Docker images; it assumes you install deps at runtime.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> int:
    return subprocess.call(cmd)


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="TutorTask103", description="Time-series forecasting CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="Run the full training pipeline (models_to_run in params.yaml)")

    p_tune = sub.add_parser("tune", help="Run Bayesian tuning script for a given model")
    p_tune.add_argument("--model", required=True, choices=["random_forest", "xgboost", "lightgbm"])

    p_serve = sub.add_parser("serve", help="Run Flask dashboard")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--debug", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "train":
        # Equivalent: python -m src.pipeline.training_pipeline
        return _run([sys.executable, "-m", "src.pipeline.training_pipeline"])

    if args.cmd == "tune":
        script = {
            "random_forest": "scripts/tune_random_forest_bayes.py",
            "xgboost": "scripts/tune_xgboost_bayes.py",
            "lightgbm": "scripts/tune_lightgbm_bayes.py",
        }[args.model]
        return _run([sys.executable, script])

    if args.cmd == "serve":
        # Run app.py directly (no long-running processes started by this agent).
        return _run([sys.executable, "app.py", "--host", args.host, "--port", str(args.port)] + (["--debug"] if args.debug else []))

    return 2


if __name__ == "__main__":
    raise SystemExit(main())


