# AI Knowledge Management for Development Teams

Decision-aware question answering over Architecture Decision Records (ADRs):
extract each decision, link it to the code it governs, and answer "why was this
designed this way?" with citations, abstaining when no rationale is recorded.

- Master issue: gpsaggese/gpsaggese.github.io#745
- Team: Gowtham S, Surendra Kumar, Saurabh Gujar
- Spec: `research/ideas/draft.AI_Knowledge_Management_Development.md`

## Status

- The project is checked in as three PRs, split along the two lanes of the
  workflow diagram (`docs/architecture.drawio`):

| PR   | Content                                                          | Status    |
| :--- | :--------------------------------------------------------------- | :-------- |
| PR 1 | Project skeleton from the course template                        | Merged    |
| PR 2 | Offline half: build the knowledge base (this directory)          | In review |
| PR 3 | Online half: search index, router, cited answers, verifier, UI   | Next      |

- This directory holds the offline half only. It needs no API key
- It runs on 10 decision records of `eclipse-edc/Connector` at one pinned
  commit: a first check of the idea on one repository, not a study

# The Offline Half

## What It Does

- The offline half turns loose Markdown decision records into a decision graph,
  then scores how current each record looks and which code no record covers:

| Step                                                 | Module                       | Output in `data/`                      |
| :--------------------------------------------------- | :--------------------------- | :------------------------------------- |
| Read files at the pinned commit, never check out     | `gitstore.py`                | (used by every step)                   |
| Read each record: title, date, sections, references  | `parser.py`                  | `records.json`                         |
| Find which records replace which                     | `parser.py`, `relations.py`  | `supersedes` in `records.json`         |
| Link each record to the code it governs              | `linker.py`                  | `links.json`, `adr_commit_signal.json` |
| Build the graph: governs, supersedes, references     | `graph.py`                   | `graph.json`                           |
| Score staleness                                      | `staleness.py`               | `staleness.json`                       |
| Rank documentation gaps                              | `gaps.py`                    | `gap_report.json`                      |

- `offline.py` runs the steps in order and `cli.py` exposes them as
  `python -m adrqa doctor` and `python -m adrqa build`
- The formulas:
  - `stale = later commits to the linked code / (1 + later edits to the record)`
  - `gap = complexity x (1 - coverage)`, per module, where `complexity` is the
    `lizard` cyclomatic complexity of its main Java code and `coverage` is the
    share of its files that some record links to

## Layout

- `src/adrqa/`: the pipeline package (offline half only)
- `tests/`: unit tests, plus `corpus` tests that check the real build output
  against `specs/`
- `specs/`: design notes (start with `00-overview.md`), corpus measurements and
  the draft answer keys in `specs/gold/`
  - The notes describe the whole project and were written before the build:
    `04` to `07` (online pipeline, evaluation, runtime, UI) arrive with PR 3
- `eval/offline_results.md`: results of this half
- `docs/architecture.drawio`: the full architecture, both halves (open it at
  app.diagrams.net)
- `*.API.ipynb`: the `adrqa` API one building block at a time, on one record
- `*.example.ipynb`: the full build on 10 records, scored against the answer
  keys, with the no-key tuning experiments
- `*_utils.py`: tables, scoring and plots used by the two notebooks
- `Dockerfile`, `docker_*.sh`, `requirements.txt`: the course template,
  unchanged

# How to Run

## With Docker (Course Setup)

- `Dockerfile` and `requirements.txt` are the course template, unchanged
  - Following the "Python Dependencies Policy" in `docker_scripts.README.md`,
    each notebook installs the pinned packages it needs in its first cell
- Build the image and start Jupyter:
  ```bash
  > ./docker_build.sh
  > ./docker_jupyter.sh
  ```
- Run `*.API.ipynb`, then `*.example.ipynb`
  - The first run clones EDC into `.cache/` (about 70 MB, no checkout) and writes
    the JSON files to `data/`; both directories are ignored by git
- Run the tests in the container:
  ```bash
  > ./docker_bash.sh
  docker> cd /git_root/class_project/msml610/Fall2026/projects/UmdTask745_MSML610_Fall2026_AI_Knowledge_Management_Development
  docker> pip install -q -e ".[test]"
  docker> python -m adrqa doctor
  docker> python -m adrqa build
  docker> python -m pytest
  ```
- Expected:
  - `doctor` passes without a key
  - `build` ends with
    `records: 10, links: 128, staleness: 8 scored, 2 null, gap modules: 20`
  - `pytest` reports `21 passed`; the 7 `corpus` tests are skipped until `build`
    has run

## Without Docker

- You need Python 3.11 or newer and `git`
- `requirements.lock` pins every package of an environment where all tests
  pass:
  ```bash
  > python -m venv .venv
  > .venv/bin/python -m pip install -r requirements.lock
  > .venv/bin/python -m pip install -e . --no-deps
  > .venv/bin/python -m adrqa doctor
  > .venv/bin/python -m adrqa build
  > .venv/bin/python -m pytest
  ```
- On Windows use `.venv/Scripts/python.exe` instead of `.venv/bin/python`

## Optional: the Model Relation Extractor

- Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and `OPENAI_CHAT_MODEL`
  - Never commit `.env`: it is ignored by git
- Install the client with `pip install -e ".[llm]"`
- `build` then also asks the model which records replace which (see below)
- The tests use a fake model and never need a key

# Results

- Full numbers are in `eval/offline_results.md`; the answer keys in
  `specs/gold/` are drafts by one person and need a second reviewer
- **Reading records**: titles, dates, the three sections, the 7 references and
  the 1 replacement are all correct, but all 10 records use one template
- **Linking records to code**: 128 links, precision 0.56 (72 of 128), recall
  0.96 (72 of 75)
  - Switching one signal off at a time (`*.example.ipynb`, Cell 3.2):

| Setting                  | Precision | Recall |
| :----------------------- | :-------- | :----- |
| All signals              | 0.56      | 0.96   |
| Without `explicit_class` | 0.40      | 0.29   |
| Without `explicit_module` | 0.74      | 0.85   |
| Without `explicit_path`  | 0.51      | 0.79   |
| Without `explicit_file`  | 0.56      | 0.95   |
| Without `adr_commit`     | 0.56      | 0.96   |

- **Staleness**: 8 records scored, 2 null (no linked code); 3 flagged at the
  default threshold of 15, 7 at 5, 5 at 10, 2 at 25
- **Gaps**: the top module is `spi/control-plane-spi` (complexity 1754,
  coverage 0.091)

# Design Choices and Tuning Knobs

## Design Choices

- **Pinned commit**: the repository is read at one SHA, so every run sees the
  same files even when the upstream repository moves on
- **`git` instead of PyDriller and a code tree**: `GitStore` reads git objects
  directly (`git show`, `git cat-file --batch`, `git log --follow`)
  - It never checks out, which avoids losing the ~100 EDC paths longer than the
    Windows path limit
- **Rules first, model optional**: `parser.py` finds replacements only in EDC's
  own wording ("supersedes" plus a link), and on six other ADR repositories
  (117 records) it found 0 of 7
  - With a key, `relations.py` asks a chat model, and keeps a claim only if its
    target is a real record and its evidence quote appears word for word in the
    record
  - Model edges are merged with rule edges, so the model can never remove one
- **One link kind**: every link is `governs`; the signal that produced it is
  kept on the link so each signal can be scored on its own
- **Null, not zero**: a record with no linked code gets no staleness score,
  because 0 would read as "fresh" when there is nothing to measure

## Tuning Knobs

| Knob                              | Now                                         | Where                     | What to try                                    |
| :-------------------------------- | :------------------------------------------ | :------------------------ | :--------------------------------------------- |
| Relation extractor model          | chat model from `.env`                      | `OPENAI_CHAT_MODEL`       | Smaller or larger model on the 117 records     |
| Extractor input limit             | 9,000 characters per record                 | `relations.py` `MAX_CHARS`| Longer, if records get cut                     |
| Extractor prompt                  | explicit replacements only, no "amends"     | `relations.py` `SYSTEM`   | Other wording, re-run on the 6 repositories    |
| Partial vs full on disagreement   | partial wins                                | `relations.py`            | Full wins, or let the model decide             |
| Linker signals                    | class, module, path, file name, ADR commit  | `linker.py`               | Drop `explicit_module` (see Results)           |
| Shared config left out of churn   | 4 build files                               | `linker.py` `SHARED_CONFIG`| Add or remove files                           |
| Stale warning threshold           | 15                                          | `STALE_WARN_THRESHOLD`    | 5, 10, 25 (swept in the example notebook)      |
| Gap list length                   | top 20                                      | `gaps.py` `rank_gaps()`   | Shorter or longer                              |

# Known Limits and Next Steps

## Known Limits

- **Linking precision**: names mentioned in passing and sentences like "this
  will remain unchanged" create wrong links, and test fixtures can be linked
  when a class name matches
- **Staleness is a hint**: it counts every later commit, including the ones that
  carried the decision out; the threshold of 15 is untested
- **Gap coverage**: it counts only the 10 records used, out of 113 in the
  repository
- **Commit signal**: on EDC the commit that added a record touches only the
  record, so this signal adds no links
- **Design notes**: `specs/` was written before the relation extractor; see the
  note at the top of `specs/03-offline-pipeline.md`

## Next Steps

- PR 3: search index (BM25 plus embeddings with reciprocal rank fusion), model
  router, graph expansion, cited answers with a citation verifier, web UI and
  the B0 to B4 evaluation
- Update the graph for new commits instead of rebuilding at one pinned SHA
- A second teammate reviews the answer keys in `specs/gold/`
