set -euo pipefail
IMG=airline-delay:latest
docker build -t $IMG .
echo "Built $IMG"
