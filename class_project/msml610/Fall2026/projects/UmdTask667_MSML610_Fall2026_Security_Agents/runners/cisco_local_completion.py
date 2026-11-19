#!/usr/bin/env python
"""Run pinned local llama.cpp, caching outputs and preserving native timings."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cisco_quick_lib import parse_timings


def main():
    binary = os.environ["CISCO_LLAMA_BINARY"]
    arguments = sys.argv[1:]
    if "--version" in arguments:
        return subprocess.call([binary, "--version"])
    # Cisco passes HF identifiers; use the verified local file instead.
    forwarded = []
    index = 0
    while index < len(arguments):
        if arguments[index] in ("--hf-repo", "--hf-file"):
            index += 2
            continue
        forwarded.append(arguments[index])
        index += 1
    identity = {
        "arguments": forwarded,
        "model_sha256": os.environ["CISCO_MODEL_SHA256"],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    cache = Path(os.environ["CISCO_LOCAL_CACHE"])
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{digest}.json"
    if path.exists():
        record = json.loads(path.read_text())
    else:
        started = time.monotonic()
        completed = subprocess.run(
            [binary, "--model", os.environ["CISCO_MODEL_FILE"]] + forwarded,
            text=True, capture_output=True, check=False, timeout=540,
        )
        record = {
            "stdout": completed.stdout, "stderr": completed.stderr,
            "returncode": completed.returncode,
            "seconds": round(time.monotonic() - started, 3),
            **parse_timings(completed.stderr),
        }
        output = (
            path if completed.returncode == 0 else path.with_suffix(".error.json")
        )
        output.write_text(json.dumps(record, indent=2) + "\n")
    sys.stdout.write(record["stdout"])
    sys.stderr.write(record["stderr"])
    return record["returncode"]


if __name__ == "__main__":
    sys.exit(main())
