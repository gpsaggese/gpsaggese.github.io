set -euo pipefail
IMG=airline-delay:latest
docker run --rm -it -p 8888:8888 -v "$(pwd)":/app $IMG \
  bash -lc 'micromamba run -n airline-delay-prediction python -m ipykernel install --user --name airline-delay-prediction && \
            micromamba run -n airline-delay-prediction jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token="" --NotebookApp.password=""'
