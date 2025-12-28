#!/usr/bin/env bash
set -euo pipefail
apt-get update
apt-get install -y --no-install-recommends ca-certificates locales tzdata wget unzip less
rm -rf /var/lib/apt/lists/*
