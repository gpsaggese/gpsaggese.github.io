#!/usr/bin/env python
"""Grade saved $10 warm-up observations with the frozen private answers."""

import argparse
import dataclasses
import json
import os
from typing import Optional, Sequence

import research.Noesis.monitor.monitor_controlled_warmup as rnmoncowar
import research.Noesis.monitor.monitor_warmup_benchmark as rnmonwaben


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank_path", required=True)
    parser.add_argument("--graders_path", required=True)
    parser.add_argument(
        "--observations_path", required=True, action="append"
    )
    parser.add_argument("--dst_dir", required=True)
    return parser.parse_args(argv)


def _load_observations(path: str):
    observations = []
    with open(path, encoding="utf-8") as file_:
        for line_number, line in enumerate(file_, start=1):
            if not line.strip():
                continue
            try:
                observations.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    "invalid observation row %s: %s" % (line_number, error)
                ) from error
    return observations


def _write_durable(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file_:
        file_.write(content)
        file_.flush()
        os.fsync(file_.fileno())


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    bank = rnmoncowar.load_warmup_bank(args.bank_path)
    configs = rnmonwaben.load_grader_configs(args.graders_path)
    rnmonwaben.validate_grader_configs(bank, configs)
    observations = []
    for observations_path in args.observations_path:
        observations.extend(_load_observations(observations_path))
    grades = tuple(
        rnmonwaben.grade_observation(observation, configs)
        for observation in observations
    )
    if len(grades) != 240:
        raise ValueError("expected exactly 240 warm-up observations")
    grades_path = os.path.join(args.dst_dir, "grades_v2.jsonl")
    summary_path = os.path.join(args.dst_dir, "grade_summary_v2.json")
    _write_durable(
        grades_path,
        "".join(
            json.dumps(dataclasses.asdict(grade), sort_keys=True) + "\n"
            for grade in grades
        ),
    )
    _write_durable(
        summary_path,
        json.dumps(
            rnmonwaben.summarize_grades(grades), indent=2, sort_keys=True
        )
        + "\n",
    )
    print("Grades: %s" % grades_path)
    print("Summary: %s" % summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
