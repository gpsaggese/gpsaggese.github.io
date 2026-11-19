#!/usr/bin/env python
"""Build the frozen 60-item warm-up bank and private grader configuration."""

import argparse
import os
from typing import Optional, Sequence

import research.Noesis.monitor.monitor_controlled_warmup as rnmoncowar
import research.Noesis.monitor.monitor_experiment_manifest as rnmonexma
import research.Noesis.monitor.monitor_warmup_benchmark as rnmonwaben


DEFAULT_BANK_PATH = "research/Noesis/data/model_drift_warmup_bank_v1.jsonl"
DEFAULT_GRADERS_PATH = (
    "research/Noesis/data/model_drift_warmup_grader_configs_v1.jsonl"
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank_path", default=DEFAULT_BANK_PATH)
    parser.add_argument("--graders_path", default=DEFAULT_GRADERS_PATH)
    return parser.parse_args(argv)


def _write_frozen(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as file_:
            if file_.read() == content:
                return
        raise ValueError("refusing to replace different frozen file %s" % path)
    with open(path, "w", encoding="utf-8", newline="") as file_:
        file_.write(content)
        file_.flush()
        os.fsync(file_.fileno())


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    bank, configs = rnmonwaben.build_bank_and_configs()
    protocol = rnmoncowar.WarmupProtocol()
    manifest = rnmonexma.DEFAULT_PHASE_ZERO_MANIFEST
    rnmoncowar.validate_warmup_bank(bank, manifest, protocol)
    rnmonwaben.validate_grader_configs(
        bank, {config.item_id: config for config in configs}
    )
    calls = rnmoncowar.build_warmup_calls(bank, manifest, protocol)
    _write_frozen(args.bank_path, rnmonwaben.render_bank_jsonl(bank))
    _write_frozen(
        args.graders_path, rnmonwaben.render_grader_configs_jsonl(configs)
    )
    print("Frozen bank: %s" % args.bank_path)
    print("Private graders: %s" % args.graders_path)
    print("Bank hash: %s" % rnmoncowar.warmup_bank_hash(bank))
    print("Validated provider calls: %s" % len(calls))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
