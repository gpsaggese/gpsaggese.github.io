#!/usr/bin/env python
"""
Download the EvalPlus datasets and the public EvalPlus samples.

The data comes from the GitHub releases of EvalPlus:
- HumanEval+ v0.1.10 and MBPP+ v0.2.0, the versions that `evalplus==0.3.1`
  uses, saved where EvalPlus looks for them, so scoring works offline
- The code that EvalPlus generated in 2023 with many models:
  - The greedy samples of every model (1 per task), for both benchmarks
  - 200 samples per task at temperatures 0.2 and 0.8 for 8 models, and at 0.8
    for ChatGPT, on HumanEval+

The zips are only downloaded here: their code is read and run by
`score_coding_runs.py` inside Docker. The size and the SHA-256 of every file
are saved in `--manifest_file`.

# Usage Example

- Download everything into `data/raw`:
> fetch_evalplus_data.py

Import as:

import fetch_evalplus_data as feevda
"""

import argparse
import concurrent.futures
import gzip
import hashlib
import logging
import os
import re
from typing import Dict, List

import pandas as pd
import requests

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser
import helpers.hretry as hretry

_LOG = logging.getLogger(__name__)

# Releases with the datasets, and the file names that EvalPlus expects.
_DATASETS = {
    "HumanEvalPlus-v0.1.10.jsonl": "https://github.com/evalplus/"
    "humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz",
    "MbppPlus-v0.2.0.jsonl": "https://github.com/evalplus/mbppplus_release/"
    "releases/download/v0.2.0/MbppPlus.jsonl.gz",
}
# Releases with the samples: v0.1.0 for HumanEval+, v0.2.0 for MBPP+.
_SAMPLE_TAGS = ("v0.1.0", "v0.2.0")
# Models whose 200 samples per task at temperatures 0.2 and 0.8 are used.
_SAMPLED_MODELS = (
    "code-llama-7b",
    "code-llama-13b",
    "code-llama-34b",
    "phind-code-llama-34b-v2",
    "starcoder",
    "codegen-16b",
    "mistral-7b",
    "codet5p-16b",
)


# #############################################################################


@hretry.sync_retry(
    (requests.RequestException,), num_attempts=5, retry_delay_in_sec=5
)
def _download(url: str, dst: str) -> None:
    """
    Download a file, retrying a few times since GitHub can drop connections.

    :param url: URL to download
    :param dst: where to save it
    """
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(dst, "wb") as file:
            for chunk in response.iter_content(chunk_size=1 << 20):
                file.write(chunk)


def _get_sample_assets() -> List[Dict[str, str]]:
    """
    List the sample zips to download from the EvalPlus releases.

    :return: 1 dict per zip, with `tag`, `name`, and `url`
    """
    assets = []
    for tag in _SAMPLE_TAGS:
        url = (
            f"https://api.github.com/repos/evalplus/evalplus/releases/tags/{tag}"
        )
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        for asset in response.json()["assets"]:
            name = asset["name"]
            # Names look like `code-llama-7b_temp_0.8.zip` or `..._temp_0.zip`.
            match = re.match(
                r"^(?P<model>.+)_temp_(?P<temp>[0-9.]+)\.zip$", name
            )
            if not match:
                continue
            temperature = float(match.group("temp"))
            is_greedy = temperature == 0
            is_sampled = tag == "v0.1.0" and (
                (
                    match.group("model") in _SAMPLED_MODELS
                    and temperature in (0.2, 0.8)
                )
                or name == "chatgpt_temp_0.8.zip"
            )
            if is_greedy or is_sampled:
                assets.append(
                    {
                        "tag": tag,
                        "name": name,
                        "url": asset["browser_download_url"],
                    }
                )
    return assets


def _fetch_asset(asset: Dict[str, str], dst_dir: str) -> Dict[str, str]:
    """
    Download 1 sample zip, unless it is already there.

    :param asset: 1 dict from `_get_sample_assets()`
    :param dst_dir: dir with 1 subdir per release
    :return: the asset with its `bytes` and `sha256`
    """
    path = os.path.join(dst_dir, asset["tag"], asset["name"])
    if not os.path.exists(path):
        hio.create_dir(os.path.dirname(path), incremental=True)
        _download(asset["url"], path)
    with open(path, "rb") as file:
        digest = hashlib.sha256(file.read()).hexdigest()
    row = dict(asset, bytes=os.path.getsize(path), sha256=digest)
    return row


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "--raw_dir",
        action="store",
        default="data/raw",
        help="Where to save the downloads",
    )
    parser.add_argument(
        "--manifest_file",
        action="store",
        default="data/evalplus_downloads.csv",
        help="Where to save the list of files with their checksums",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Save the datasets where EvalPlus looks for them (see
    # `docker_cmd_sandbox.sh`).
    cache_dir = os.path.join(args.raw_dir, "evalplus_cache", "evalplus")
    hio.create_dir(cache_dir, incremental=True)
    rows = []
    for name, url in _DATASETS.items():
        gz_path = os.path.join(cache_dir, os.path.basename(url))
        if not os.path.exists(gz_path):
            _download(url, gz_path)
        with gzip.open(gz_path, "rb") as file:
            hio.to_file(
                os.path.join(cache_dir, name), file.read().decode("utf-8")
            )
        with open(gz_path, "rb") as file:
            digest = hashlib.sha256(file.read()).hexdigest()
        rows.append(
            {
                "tag": "dataset",
                "name": os.path.basename(url),
                "url": url,
                "bytes": os.path.getsize(gz_path),
                "sha256": digest,
            }
        )
    # Download the sample zips, 3 at a time.
    samples_dir = os.path.join(args.raw_dir, "evalplus_samples")
    assets = _get_sample_assets()
    _LOG.info("Downloading '%s' sample zips", len(assets))
    with concurrent.futures.ThreadPoolExecutor(3) as executor:
        rows.extend(executor.map(lambda a: _fetch_asset(a, samples_dir), assets))
    # Save the list of files with their checksums.
    downloads = pd.DataFrame(rows).sort_values(["tag", "name"])
    downloads.to_csv(args.manifest_file, index=False)
    _LOG.info(
        "Saved '%s' files ('%.0f' MB)",
        len(downloads),
        downloads["bytes"].sum() / 1e6,
    )


if __name__ == "__main__":
    _main(_parse())
