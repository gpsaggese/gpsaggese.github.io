#!/usr/bin/env python3
"""
Download the bake-off benchmarks, fix the task sample, and check the scorers.

Usage:
> download_benchmarks.py --download
> download_benchmarks.py --sample 50
> download_benchmarks.py --check

`--download` puts the raw data in `data/benchmarks/` (not committed):
- InfiAgent-DABench, QRData, DA-Code: sparse git clones from GitHub
- DA-Code task data: `source.zip` from Google Drive with `gdown`; the GitHub repo
  holds only 62 of the 500 task folders. If Google Drive refuses the scripted
  download, get the zip in a browser and pass `--dacode-zip <path>`
- DABstep: `huggingface_hub.snapshot_download` of `adyen/DABstep`

`--sample` writes `benchmark_sample.json`, the fixed list of task ids every
agent runs on. It is committed, so all runs use the same tasks.

`--check` verifies, for every sampled task, that the data files exist, that the
gold answer scores 1, and that an empty answer scores 0.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from typing import List, Optional

import helpers.hdbg as hdbg
import helpers.hparser as hparser

import research.agentic_data_science.bakeoff.benchmarks as aebbench

_LOG = logging.getLogger(__name__)

# Repo, commit-independent sparse paths.
_GIT_SOURCES = {
    "InfiAgent": (
        "https://github.com/InfiAgent/InfiAgent.git",
        ["examples/DA-Agent/data"],
    ),
    "QRData": ("https://github.com/xxxiaol/QRData.git", ["benchmark"]),
    "da-code": (
        "https://github.com/yiyihum/da-code.git",
        ["da_code/configs", "da_code/source", "da_code/gold"],
    ),
}


def _git_sparse_clone(url: str, dst: str, paths: List[str]) -> None:
    if os.path.isdir(os.path.join(dst, ".git")):
        _LOG.info("Already cloned: %s", dst)
        return
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", url, dst],
        check=True,
    )
    subprocess.run(["git", "-C", dst, "sparse-checkout", "set", *paths], check=True)


# `source.zip` of DA-Code, from the `gdown` command in its README.
_DACODE_SOURCE_GDRIVE_ID = "1ZGi0iNv797OneK_b2l2thJCulP6NUf7v"


def _download_dacode_source(source_dir: str, zip_path: Optional[str]) -> bool:
    """
    Add the DA-Code task folders that are not in the GitHub repo.

    :param zip_path: a `source.zip` downloaded by hand; None to fetch it with
        `gdown`
    :return: whether the task folders are complete
    """
    if len(os.listdir(source_dir)) >= 500:
        _LOG.info("DA-Code source already complete")
        return True
    with tempfile.TemporaryDirectory() as tmp:
        if zip_path is None:
            zip_path = os.path.join(tmp, "source.zip")
            try:
                import gdown

                gdown.download(id=_DACODE_SOURCE_GDRIVE_ID, output=zip_path, quiet=False)
            except Exception as e:  # pylint: disable=broad-except
                _LOG.error(
                    "Could not download DA-Code source.zip (%s). Download it in a "
                    "browser from https://drive.google.com/uc?id=%s and rerun with "
                    "--download --dacode-zip <path>",
                    type(e).__name__,
                    _DACODE_SOURCE_GDRIVE_ID,
                )
                return False
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        # Task folders sit at an undocumented depth: find them by name.
        moved = 0
        for dirpath, dirnames, _ in os.walk(tmp):
            for d in list(dirnames):
                if "-" in d and d.split("-")[0] in ("di", "dm", "dw", "ml", "sa", "plot", "data"):
                    dst = os.path.join(source_dir, d)
                    if not os.path.exists(dst):
                        shutil.move(os.path.join(dirpath, d), dst)
                        moved += 1
                    dirnames.remove(d)
        _LOG.info("Added %d DA-Code task folders", moved)
    return True


def _download(root: str, dacode_zip: Optional[str] = None) -> None:
    os.makedirs(root, exist_ok=True)
    for name, (url, paths) in _GIT_SOURCES.items():
        _LOG.info("Downloading %s", name)
        _git_sparse_clone(url, os.path.join(root, name), paths)
    # QRData ships its data files as a zip.
    qr = os.path.join(root, "QRData", "benchmark")
    if not os.path.isdir(os.path.join(qr, "data")):
        with zipfile.ZipFile(os.path.join(qr, "data.zip")) as z:
            z.extractall(qr)
    # DABstep is on Hugging Face.
    try:
        import huggingface_hub
    except ImportError:
        raise SystemExit(
            "DABstep needs `pip install huggingface_hub`; then rerun --download"
        )
    _LOG.info("Downloading DABstep")
    huggingface_hub.snapshot_download(
        repo_id="adyen/DABstep",
        repo_type="dataset",
        local_dir=os.path.join(root, "DABstep"),
    )
    # Last, because Google Drive often refuses scripted downloads.
    if not _download_dacode_source(
        os.path.join(root, "da-code", "da_code", "source"), dacode_zip
    ):
        raise SystemExit(1)


def _sample(root: str, n: int, seed: int) -> None:
    out = {}
    for name in aebbench.LOADERS:
        if name == "dabstep" and n >= 10:
            # Only the 10 dev tasks have answers, so the sample is all of them.
            out[name] = "all"
            continue
        tasks = aebbench.load_tasks(name, root)
        out[name] = aebbench.sample_ids(tasks, n, seed)
        _LOG.info("%s: %s of %d tasks sampled", name, n, len(tasks))
    with open(aebbench.SAMPLE_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"Wrote {aebbench.SAMPLE_FILE}")


def _check(root: str) -> bool:
    ok = True
    for name in aebbench.LOADERS:
        try:
            tasks = aebbench.load_sample(name, root)
        except FileNotFoundError:
            print(f"{name:8s} not downloaded: run --download  FAIL")
            ok = False
            continue
        missing = [f for t in tasks for f in t.files if not os.path.exists(f)]
        no_files = [t.task_id for t in tasks if not t.files]
        gold = sum(aebbench.score(t, aebbench.gold_as_prediction(t)) for t in tasks)
        empty = sum(aebbench.score(t, "") for t in tasks)
        good = not missing and not no_files and gold == len(tasks) and empty == 0
        ok &= good
        print(
            f"{name:8s} tasks={len(tasks):3d}  gold={gold:.0f}/{len(tasks)}  "
            f"empty={empty:.0f}  missing_files={len(missing)}  "
            f"no_files={len(no_files)}  {'OK' if good else 'FAIL'}"
        )
    return ok


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--download", action="store_true", help="Download data")
    parser.add_argument(
        "--dacode-zip", help="DA-Code source.zip downloaded by hand from Google Drive"
    )
    parser.add_argument("--sample", type=int, help="Sample N tasks per benchmark")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--check", action="store_true", help="Check the scorers")
    parser.add_argument(
        "--root", default=aebbench.BENCHMARK_DIR, help="Where the data lives"
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser, argv: Optional[List[str]] = None) -> None:
    args = parser.parse_args(argv)
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    if args.download:
        _download(args.root, args.dacode_zip)
    if args.sample:
        _sample(args.root, args.sample, args.seed)
    if args.check and not _check(args.root):
        raise SystemExit(1)
    if not (args.download or args.sample or args.check):
        parser.print_help()


if __name__ == "__main__":
    _main(_parse())
