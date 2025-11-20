#!/usr/bin/env bash
set -euo pipefail
IMG=airline-delay:latest
docker run --rm -it -p 8501:8501 -v "$(pwd)":/app $IMG \
  bash -lc 'micromamba run -n airline-delay-prediction streamlit run src/app.py --server.address=0.0.0.0 --server.port=8501'
