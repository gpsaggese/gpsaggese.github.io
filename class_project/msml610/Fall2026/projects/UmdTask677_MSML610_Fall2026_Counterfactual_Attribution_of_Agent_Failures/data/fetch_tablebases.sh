#!/usr/bin/env bash
# Download the KRvK Syzygy tables (WDL + DTZ, about 8 KB) from lichess and check
# them against its published sha256 list.
set -euo pipefail
cd "$(dirname "$0")/syzygy"
BASE=https://tablebase.lichess.ovh/tables/standard
curl -fsS -o sha256.txt "$BASE/sha256"
curl -fsS -o KRvK.rtbw "$BASE/3-4-5-wdl/KRvK.rtbw"
curl -fsS -o KRvK.rtbz "$BASE/3-4-5-dtz/KRvK.rtbz"
for f in KRvK.rtbw KRvK.rtbz; do
  want=$(grep -E " $f\$" sha256.txt | cut -d' ' -f1)
  got=$(shasum -a 256 "$f" | cut -d' ' -f1)
  [ "$want" = "$got" ] && echo "$f ok" || { echo "$f checksum mismatch" >&2; exit 1; }
done
