# Run the Checker Replay Demo

This guide reproduces the offline checker-validation result used by PR_S6. The
demo is deterministic and requires no API key, network call, provider, database,
or market process.

## Run the Default Experiment

From the repository root:

```bash
research/Noesis/scripts/run_checker_replay.py
```

The command runs 30,000 simulated contracts and prints a Markdown summary. It
writes these files under `research/Noesis/docs/findings/checker_replay/`:

- `checker_replay.md`: concise findings for a reviewer
- `checker_replay.csv`: per-seed and aggregate rows for analysis
- `checker_replay.json`: complete machine-readable protocol and results

Exit code zero means every scenario met both the statistical tolerance and the
five-percentage-point roadmap tolerance. A nonzero exit means at least one
scenario did not agree with the analytic prediction.

## Run a Fast Review Demo

Use a temporary output directory so the checked-in findings are not replaced:

```bash
research/Noesis/scripts/run_checker_replay.py \
    --seeds 0,1 \
    --contracts_per_seed 100 \
    --output_dir /tmp/noesis_checker_replay
```

The smaller run is useful for reviewing the flow. The checked-in scientific
result uses the full default protocol.

## Run the Focused Tests

```bash
pytest -q \
    research/Noesis/checker/test/test_checker_replay.py \
    research/Noesis/checker/test/test_checker_replay_runner.py \
    research/Noesis/checker/test/test_checker_replay_report.py \
    research/Noesis/scripts/test/test_run_checker_replay.py
```

## Check Reproducibility

Run the full command twice with two different temporary output directories, then
compare the artifacts:

```bash
diff -r /tmp/noesis_checker_replay_1 /tmp/noesis_checker_replay_2
```

An empty diff confirms that the JSON, CSV, and Markdown outputs are byte-for-byte
reproducible for the same protocol.

## Interpret the Demo

- `healthy` checks false alarms.
- `slow_only` must fail only on latency.
- `weak_model_only` must fail only on quality.
- The two `borderline_*` scenarios test finite-sample uncertainty.
- `insufficient_evidence` must close at the deadline without inventing a pass or
  seller failure.

The demo validates checker software and statistics. It is not a live-provider
measurement and cannot by itself establish model drift or nerfing.
