# Codebase Setup Plan

- Goal: set up `AI_Security_Agents` so that it:
  - Holds the labelled Python security benchmark
  - Runs 3 open-source LLM agents plus 2 non-LLM baselines on every snippet
  - Writes every finding in one normalised format
  - Scores findings against the labels
- The code is written to be copied unchanged into the course repo later
  - Target: `umd_classes/class_project/msml610/Fall2026/projects/UmdTask667_MSML610_Fall2026_Security_Agents/`
  - Issue: `gpsaggese/umd_classes#667`
- Cost: $0
  - Agents are open source and installed locally
  - Models come from free API tiers; a local model is a backup only

## Ground Rules

- AI assistant:
  - Creates and edits files in this repo only
  - Never runs `git commit`, `git push`, or any GitHub command
  - Never reads `.env` or prints API keys
  - Asks before installing anything outside the project venv
- Dakshi:
  - Does all git and GitHub activity
  - Creates API keys and puts them in `.env`
- Every step ends with a check that can be run
  - The next step starts only after that check passes

## Decisions

| Topic | Decision | Reason |
| :---- | :------- | :----- |
| Python | 3.11 venv via `uv` | Course lint target is `py311` |
| Code style | Course coding conventions | Move without rewrite |
| Helpers | Import course `helpers` via `PYTHONPATH` | Same code works in both repos |
| Line length | 81 | Course `pyproject.toml` |
| Main model | `openai/gpt-oss-120b` on Groq free tier | Free, open-weight, OpenAI-compatible |
| Backup model | Gemini free tier, then local `gpt-oss:20b` | Used when Groq rate limits are hit |
| OWASP list | Top 10:2025 | Final and current |
| Core agents | Vulnhuntr, Cisco `ai-deep-sast`, Codex CLI | Open source, accept any OpenAI-compatible API |
| Fallback agent | OpenHands | Use if Codex CLI cannot talk to Groq |
| Baselines | Semgrep CE, Bandit | Free, no API, SARIF output |

## Target Layout

```verbatim
AI_Security_Agents/
├── README.md
├── plan.md                      this file
├── .gitignore
├── .env.example                 key names only, no values
├── requirements.txt
├── setenv.sh                    activates venv, sets PYTHONPATH to course helpers
├── benchmark/
│   ├── seeded/                  SEED-001.py, ...
│   ├── secure/                  SAFE-001.py, ...
│   ├── cve/                     CVE-YYYY-NNNNN_vuln.py, CVE-YYYY-NNNNN_fixed.py
│   ├── multifile/               TAINT-001/{api.py,service.py,db.py}, ...
│   ├── exploit_tests/           test_SEED-001.py, ...
│   └── labels.csv               single source of truth for labels
├── benchmark_lib.py             parse headers, load labels, validate, strip answers
├── findings_lib.py              normalised finding schema, SARIF reader
├── scoring_lib.py               match findings to labels, compute metrics
├── runners/
│   ├── run_semgrep.py
│   ├── run_bandit.py
│   ├── run_plain_llm.py
│   ├── run_vulnhuntr.py
│   ├── run_ai_deep_sast.py
│   └── run_codex.py
├── results/
│   ├── raw/                     untouched agent output (gitignored)
│   └── normalized/              one JSONL per agent and run
└── test/
    ├── test_benchmark_lib.py
    ├── test_findings_lib.py
    └── test_scoring_lib.py
```

## Step 1: Repo Skeleton

- Create:
  - Folders from the layout above, with `.gitkeep` in empty ones
  - `.gitignore`:
    - `.env`
    - `.venv/`
    - `results/raw/`
    - `third_party/`
    - `__pycache__/`
    - `*.gguf`
    - `.ipynb_checkpoints/`
  - `.env.example` with empty `GROQ_API_KEY=` and `GEMINI_API_KEY=`
  - `README.md` sections: goal, setup, how to run, label schema
- Check:
  ```bash
  > git status --short
  ```
  - `.env` and `results/raw/` never appear

## Step 2: Python Environment

- Create the venv and install the base packages:
  ```bash
  > uv venv --python 3.11 .venv
  > source .venv/bin/activate
  > uv pip install -r requirements.txt
  ```
- `requirements.txt`:
  - `pandas`
  - `pytest`
  - `python-dotenv`
  - `openai`
  - `semgrep`
  - `bandit`
  - `tqdm`
- `setenv.sh`:
  - Activates `.venv`
  - Adds `../umd_classes/helpers_root` to `PYTHONPATH`
- Check:
  ```bash
  > source setenv.sh
  > python -c "import helpers.hdbg, pandas, openai; print('ok')"
  > semgrep --version
  > bandit --version
  ```

## Step 3: Benchmark Schema and Validator

- Snippet header (docstring at the top of every snippet):
  ```python
  """
  id: SEED-001
  type: seeded
  vulnerable: true
  cwe: 89
  owasp: A05:2025 Injection
  severity: high
  bug_lines: [12]
  attack_scenario: Attacker sends ' OR '1'='1 as email and dumps all users
  remediation: Use a parameterised query
  source: Written by us, based on the CWE-89 page
  cve_id:
  fix_commit:
  pair_id: P-001
  labelled_by: Dakshi
  reviewed_by:
  """
  ```
  - Mark each bug line in code with `# {!LINE}`
- `labels.csv` columns:
  - `snippet_id`, `type`, `vulnerable`, `cwe`, `owasp_2025`, `severity`,
    `bug_lines`, `attack_scenario`, `remediation`, `cve_id`, `fix_commit`,
    `pair_id`, `exploit_test`, `labelled_by`, `reviewed_by`, `first_pass_agree`
- `benchmark_lib.py` functions:
  - `parse_header(path)`: read the docstring into a dict
  - `load_labels(csv_path)`: read `labels.csv` into a dataframe
  - `validate_benchmark(benchmark_dir)`: check that:
    - Every snippet has a header
    - Header fields match its `labels.csv` row
    - A vulnerable snippet has a `# {!LINE}` marker on each line in `bug_lines`
    - A safe snippet has no marker and an empty `bug_lines`
    - Snippets are 15-80 lines
    - Every `pair_id` has exactly 2 members
  - `strip_answers(path) -> (code, line_map)`:
    - Removes the header docstring and `# {!LINE}` markers
    - Returns a map from stripped line numbers to original line numbers
    - Agents only ever see stripped code
- Check:
  ```bash
  > pytest test/test_benchmark_lib.py
  ```
  - Tests cover one valid snippet and one broken snippet per rule

## Step 4: Prototype Snippets

- Write the 8 key examples from the project brief as 4 pairs:

  | Pair | Vulnerable | Safe twin | CWE |
  | :--- | :--------- | :-------- | :-- |
  | SQL injection | f-string query | parameterised query | 89 |
  | XSS | unescaped input in HTML | `html.escape` | 79 |
  | Deserialisation | `pickle.loads` on request data | `json.loads` | 502 |
  | Path traversal | `os.path.join` with no check | `realpath` checked against base | 22 |

- Rules for every snippet:
  - Small Flask route, 15-60 lines, Python 3
  - User input and risky line in the same file
  - Own wording, not copied from public examples
- Fill `labels.csv`
  - Leave `reviewed_by` empty until a teammate reviews it blind
- Check:
  ```bash
  > python -c "import benchmark_lib as b; b.validate_benchmark('benchmark')"
  ```

## Step 5: Normalised Findings

- `findings_lib.py` defines one finding record:
  - `snippet_id`, `agent`, `model`, `run_id`
  - `line`, `cwe`, `severity`, `confidence`
  - `explanation`, `fix`
  - `tokens_in`, `tokens_out`, `n_requests`, `seconds`
- Functions:
  - `read_sarif(path, agent)`: SARIF file to finding records
  - `write_findings(findings, path)`: append to a JSONL file
  - `remap_lines(findings, line_map)`: map stripped line numbers back to the
    original snippet
- Check:
  ```bash
  > pytest test/test_findings_lib.py
  ```
  - Uses a small saved SARIF sample

## Step 6: Baseline Runners (No API)

- `run_semgrep.py`:
  - Config: `p/python` and `p/owasp-top-ten`
  - Output: SARIF to `results/raw/semgrep/`, then normalised JSONL
- `run_bandit.py`:
  - Output: SARIF or JSON to `results/raw/bandit/`, then normalised JSONL
- Every runner:
  - Uses the course script structure (`_parse()` and `_main(parser)`)
  - Works on stripped copies of the snippets in a temp dir
  - Records the tool version
- Check:
  ```bash
  > python runners/run_semgrep.py --benchmark_dir benchmark --run_id r1
  > python runners/run_bandit.py --benchmark_dir benchmark --run_id r1
  ```
  - Both write findings for the 8 prototypes

## Step 7: Scoring on the Prototype

- `scoring_lib.py`:
  - Match rule: a finding is a true positive if:
    - It is on a vulnerable snippet
    - Its line is within the labelled bug lines +-3
    - Its CWE equals the label CWE or a parent/child CWE
  - Snippet-level verdict: flagged if any finding exists
  - Metrics per agent:
    - Precision, recall, F1
    - FPR on safe snippets: FP / (FP + TN)
    - False discovery rate: FP / (FP + TP)
    - MCC
    - Paired accuracy over `pair_id`
- Check:
  ```bash
  > pytest test/test_scoring_lib.py
  ```
  - Prints one metrics table for Semgrep and Bandit
- Milestone:
  - End-to-end pipeline works with no LLM yet
  - Good point for Dakshi to commit

## Step 8: Free LLM Access

- Dakshi:
  - Creates a Groq API key
  - Copies `.env.example` to `.env` and fills `GROQ_API_KEY`
- `run_plain_llm.py`:
  - Sends each stripped snippet with one fixed prompt to `openai/gpt-oss-120b`
  - Asks for JSON findings
  - Caches every raw response in `results/raw/plain_llm/`
  - Retries with backoff on HTTP 429
  - Logs tokens and requests
- Check:
  ```bash
  > python runners/run_plain_llm.py --benchmark_dir benchmark --run_id r1
  ```
  - 8 cached responses and a normalised JSONL

## Step 9: Agent Runners

- Rule: at most 1 day per agent; switch to the fallback if it fails
- Vulnhuntr:
  - Install in its own Python 3.10 environment:
    ```bash
    > uv tool install --python 3.10 git+https://github.com/protectai/vulnhuntr.git
    ```
  - Settings:
    - `OPENAI_BASE_URL=https://api.groq.com/openai/v1`
    - `OPENAI_MODEL=openai/gpt-oss-120b`
    - `OPENAI_API_KEY` from `GROQ_API_KEY`
  - Runner calls `vulnhuntr -r <dir> -l gpt` and parses its report
- Cisco `ai-deep-sast`:
  - Clone into `third_party/ai-deep-sast` (gitignored) and install its
    requirements
  - Settings: `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`
  - Runner calls `deepscan.py --target <dir>` and parses its JSON
- Codex CLI with Trail of Bits skills:
  - Install `@openai/codex` (needs Node) and the `trailofbits/skills` plugin
  - Add a Groq provider in `~/.codex/config.toml`
  - Runner calls `codex exec` with a fixed review prompt that asks for JSON
  - Fallback: OpenHands with the same model and prompt
- Check for each agent:
  - Runs on 2 prototype snippets
  - Produces parseable findings
  - Records model, tokens, and seconds

## Step 10: Move to the Course Repo

- Checklist:
  - [ ] GP has approved the plan on issue #667
  - [ ] Copy files into the course project folder, excluding:
    - `.git/`
    - `.env`
    - `.venv/`
    - `results/raw/`
    - `third_party/`
  - [ ] Keep the course folder's Docker files
  - [ ] Merge `requirements.txt` into the course folder's `requirements.txt`
  - [ ] Drop `setenv.sh`: the course repo provides `helpers` directly
  - [ ] Run `pytest` inside the course Docker container
  - [ ] Dakshi commits on branch `UmdTask667_MSML610_Fall2026_Security_Agents`
        with suffix `(gpsaggese/umd_classes#667)`

## Progress

- [ ] Step 1: Repo skeleton
- [ ] Step 2: Python environment
- [ ] Step 3: Benchmark schema and validator
- [ ] Step 4: Prototype snippets
- [ ] Step 5: Normalised findings
- [ ] Step 6: Baseline runners
- [ ] Step 7: Scoring on the prototype
- [ ] Step 8: Free LLM access
- [ ] Step 9: Agent runners
- [ ] Step 10: Move to the course repo
