set -euo pipefail
IMG=airline-delay:latest
# Mount repo into /app and keep notebooks/outputs on host
docker run --rm -it -p 8501:8501 -p 8888:8888 -v "$(pwd)":/app $IMG bash
