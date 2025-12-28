"""
Minimal Flask dashboard to inspect latest run metrics + plots.

Run (inside Docker):
  python app.py --host 0.0.0.0 --port 8000
Then open:
  http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flask import Flask, render_template, send_from_directory, redirect, url_for, request


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_PATH = ARTIFACTS_DIR / "metrics" / "last_run.json"


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"))

    @app.get("/")
    def index():
        metrics = None
        if METRICS_PATH.exists():
            metrics = json.loads(METRICS_PATH.read_text())
        plots = sorted([p.name for p in PLOTS_DIR.glob("*.png")]) if PLOTS_DIR.exists() else []
        return render_template("index.html", metrics=metrics, plots=plots)

    @app.get("/plots/<path:filename>")
    def plots(filename: str):
        return send_from_directory(PLOTS_DIR, filename)

    @app.post("/refresh")
    def refresh():
        # This only refreshes the page; running training should be done via main.py / docker command.
        return redirect(url_for("index"))

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()


