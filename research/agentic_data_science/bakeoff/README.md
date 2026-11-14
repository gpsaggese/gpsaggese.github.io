# Agentic EDA Bake-off

This harness runs several EDA agents over the same datasets and records the
results in one scorecard. It is part of the Agentic Data Science project; see
`../docs/plan.md`. It was first merged in `gpsaggese/tutorials#706` and moved
here when that repo was archived.

Each existing agent, plus a couple of external baselines, gets wrapped behind
one interface. The harness then runs all of them on a fixed set of datasets and
produces a single table that compares them.

## Usage

From the repo root (inside the dev container):

```bash
python research/agentic_data_science/bakeoff/run_bakeoff.py --list
python research/agentic_data_science/bakeoff/run_bakeoff.py --run
python research/agentic_data_science/bakeoff/run_bakeoff.py --run --agent dummy --dataset uci_heart
pytest research/agentic_data_science/bakeoff/test
```

`--run` writes `results/scorecard.csv`.

## Layout

| File | Contents |
|---|---|
| `datasets.py` | `Dataset` and the registry of the four datasets |
| `agents.py` | `Agent` interface, `AgentResult`, `DummyAgent` |
| `scoring.py` | `Scorecard` / `ScoreRow`, CSV output |
| `run_bakeoff.py` | CLI |
| `benchmarks.py` | Benchmark loaders and scorers |
| `dabstep_scorer.py` | Official DABstep scorer (vendored) |
| `download_benchmarks.py` | Download, sample and check the benchmarks |
| `benchmark_sample.json` | Fixed task sample used by every run |
| `data/` | local cache for real datasets (not committed) |
| `results/` | run outputs (not committed) |

## Interfaces

An agent implements `name` and `run(dataset) -> AgentResult`. The harness calls
`agent.timed_run(dataset)`, which times the run and turns an exception into a
recorded failure, so one broken agent does not stop the bake-off.

For each (agent, dataset) pair, the scorecard records `runs_ok`, `runtime_s`,
`n_insights` and the error. The `insight_quality` and `code_quality` columns are
already in the schema but stay blank until the scoring PR fills them.

## Datasets

The four datasets are `uci_heart` (classification), `nyc_taxi` (regression),
`openaq` (regression) and `amazon_reviews` (text). `load()` returns a small
synthetic sample with the real schema, so everything runs offline.
`load(real=True)` reads `data/<name>.csv` or a wired `real_loader`. If neither
exists it raises, naming the source.

## Benchmarks

The datasets above have no answer key. For Milestone 1 we score agents on four
public benchmarks with known answers, using each benchmark's own scoring rules
(`benchmarks.py`):

| Name    | Benchmark         | Tasks we use               | Data source                     |
| :------ | :---------------- | :------------------------- | :------------------------------ |
| dabench | InfiAgent-DABench | 256 dev questions          | GitHub                          |
| qrdata  | QRData            | 411 questions              | GitHub                          |
| dacode  | DA-Code           | 71 tasks with text answers | GitHub + Google Drive (`gdown`) |
| dabstep | DABstep           | 10 dev tasks               | Hugging Face                    |

Every agent runs on the same 50 tasks per benchmark, listed in
`benchmark_sample.json` (seed 0). DABstep has only 10 tasks with public answers,
so all 10 are used.

```bash
pip install gdown huggingface_hub
python research/agentic_data_science/bakeoff/download_benchmarks.py --download
python research/agentic_data_science/bakeoff/download_benchmarks.py --check
```

`--check` confirms, for every sampled task, that its data files exist, that the
gold answer scores 1, and that an empty answer scores 0.

Limits to keep in mind when reading results:
- DA-Code: 429 of its 500 tasks are scored on files the agent writes (CSV, plots,
  models). They are not used here
- DABstep: its 450 test answers are hidden; scoring them needs a leaderboard
  submission
- InfiAgent-DABench: question 734 is skipped because its label repeats an answer
  name with different values

## Adding an agent

1. Subclass `Agent` with a wrapper that adapts the existing agent
   (`schema_agent`, `intermediate_v0_timeseries_agent`, AgenTUI, PandasAI) to
   `run(dataset) -> AgentResult`.
2. Add it to `agents.all_agents()`.
3. Run `run_bakeoff.py --run`. The new agent appears in the scorecard.
