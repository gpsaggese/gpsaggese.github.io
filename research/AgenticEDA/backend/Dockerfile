FROM python:3.12-slim

WORKDIR /app

# System deps for numpy, scipy, matplotlib, weasyprint
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran libopenblas-dev curl \
    libgraphite2-3 libharfbuzz0b libicu-dev libfontconfig1 libfreetype6 \
    libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 libcairo2 libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Tectonic for PDF export (single binary LaTeX compiler).
#
# Installed by direct download rather than the upstream drop-installer script.
# The script derives an aarch64 target of "aarch64-unknown-linux-gnu", which
# upstream has never published — it ships a musl build for aarch64 instead — so
# the script 404s on every arm64 build. Selecting the asset by architecture
# here lets the image build on both amd64 and arm64 hosts.
ARG TECTONIC_VERSION=0.17.0
RUN set -eux; \
    case "$(dpkg --print-architecture)" in \
      amd64) tectonic_target='x86_64-unknown-linux-gnu' ;; \
      arm64) tectonic_target='aarch64-unknown-linux-musl' ;; \
      *) echo "No tectonic build for $(dpkg --print-architecture)" >&2; exit 1 ;; \
    esac; \
    curl --proto '=https' --tlsv1.2 -fsSL \
      "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40${TECTONIC_VERSION}/tectonic-${TECTONIC_VERSION}-${tectonic_target}.tar.gz" \
      -o /tmp/tectonic.tar.gz; \
    tar -xzf /tmp/tectonic.tar.gz -C /usr/local/bin tectonic; \
    rm /tmp/tectonic.tar.gz; \
    tectonic --version

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY backend/ backend/
COPY src/ src/
COPY ieee_template/ ieee_template/

# Create sessions dir
RUN mkdir -p sessions

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
