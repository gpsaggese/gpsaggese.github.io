# AI Security Agents

- Compares AI security analysis agents on a labelled Python benchmark with known
  ground truth
  - Measures detection, false alarms, localization, CWE classification,
    explanations, and fixes
  - Uses only open-source agents and free model APIs

# Project Description
- Which agent catches the most real vulnerabilities with the fewest false
  alarms, and gives clear explanations and working fixes?
- How do agents compare on seeded bugs, real CVEs, and safe code that looks
  risky?
- When an agent proposes a fix, how often does the fix remove the bug without
  breaking the code?

## Benchmark

- 62 Python snippets, each with a ground-truth label in its header

  | Pile               | Count | Vulnerable | Source                                      |
  | :----------------- | :---- | :--------- | :------------------------------------------ |
  | Seeded bugs        | 20    | Yes        | Written by us, OWASP Top 10:2025            |
  | Secure look-alikes | 18    | No         | 9 safe twins, 9 risky-looking but safe      |
  | Real CVEs, before  | 12    | Yes        | Upstream code at the parent of the fix      |
  | Real CVEs, after   | 12    | No         | Upstream code at the fix commit             |

- Seeded bugs cover 8 OWASP 2025 categories:
  - A01 Broken Access Control: 4
  - A02 Security Misconfiguration: 2
  - A04 Cryptographic Failures: 3
  - A05 Injection: 5
  - A07 Authentication Failures: 2
  - A08 Software or Data Integrity Failures: 2
  - A09 Security Logging and Alerting Failures: 1
  - A10 Mishandling of Exceptional Conditions: 1
- Real CVEs come from Django, PyJWT, Jinja, requests, urllib3, and Werkzeug
  - CWE and severity come from the GitHub Advisory Database
- Exploit tests run the attack on 31 snippets
  - The attack must succeed on the vulnerable snippet and fail on its safe twin

## Agents

| Agent                                  | Type                         | Role      |
| :------------------------------------- | :--------------------------- | :-------- |
| Codex CLI with Trail of Bits skills    | Agentic code reviewer        | Core      |
| Vulnhuntr                              | Source-to-sink LLM tracer    | Core      |
| Cisco `ai-deep-sast`                   | Static analysis plus LLM     | Core      |
| Plain LLM, one prompt                  | Same model, no agent loop    | Baseline  |
| Semgrep CE, Bandit                     | Rule-based static analysis   | Baseline  |
| Ensemble agent (LangGraph)             | Combines the agents above    | Built     |

- Model: `openai/gpt-oss-120b` on the Groq free tier
  - Backup: Gemini free tier, or a local `gpt-oss:20b` through Ollama
  - Free-tier limits: 1,000 requests and 200,000 tokens per day, 8,000 tokens
    per minute, shared by all agents using the same key
- Vulnhuntr (pinned at commit `ead88c5`) runs in its own Python 3.10 Docker
  image and is limited by design:
  - Reports 7 vulnerability types only: LFI, RCE, SSRF, AFO, SQLI, XSS, IDOR
  - Reports no line numbers, so it is scored at snippet level only (its
    localization metrics are N/A)
  - Proposes no fixes

## Metrics

- Snippet level: a snippet is flagged if the agent reports any finding on it
  - Precision, recall, F1, MCC
  - FPR: FP / (FP + TN), share of safe snippets flagged
  - FDR: FP / (FP + TP), what the brief calls "false positive rate"
  - Youden J: recall - FPR
- Finding level: a finding localizes the bug if it is within 3 lines of a
  labelled bug line
  - Strict mode also requires the CWE to match, or a related CWE
- Paired accuracy: the vulnerable snippet is flagged and its safe twin is not
- Later phases: explanation quality (1-5, blind, 3 raters), fix success rate,
  tokens and seconds per true positive

## Cumulative Results

- Snippet-level scores of every run in `results/scores.csv`, which
  `score_results.py` regenerates
  - Smoke tests (1-2 snippets) are omitted

  | Agent                 | Snippets | TP | FP | FN | TN | Precision | Recall | F1   | FPR  | MCC  |
  | :-------------------- | :------- | :- | :- | :- | :- | :-------- | :----- | :--- | :--- | :--- |
  | `ai_deep_sast`        | 16       | 3  | 1  | 5  | 7  | 0.75      | 0.38   | 0.50 | 0.13 | 0.29 |
  | `ai_deep_sast_quick`  | 62       | 15 | 6  | 17 | 24 | 0.71      | 0.47   | 0.57 | 0.20 | 0.28 |
  | `bandit`              | 62       | 12 | 7  | 20 | 23 | 0.63      | 0.38   | 0.47 | 0.23 | 0.15 |
  | `plain_llm`           | 62       | 27 | 16 | 5  | 14 | 0.63      | 0.84   | 0.72 | 0.53 | 0.34 |
  | `semgrep`             | 62       | 9  | 2  | 23 | 28 | 0.82      | 0.28   | 0.42 | 0.07 | 0.28 |
  | `semgrep_cisco_rules` | 62       | 15 | 6  | 17 | 24 | 0.71      | 0.47   | 0.57 | 0.20 | 0.28 |
  | `vulnhuntr`           | 62       | 15 | 6  | 17 | 24 | 0.71      | 0.47   | 0.57 | 0.20 | 0.28 |

- Notes on the table:
  - `ai_deep_sast` `r2_partial` covers only the 16 snippets Cisco fully
    evaluated before the Groq daily quota ran out: do not compare it with the
    62-snippet rows
  - `semgrep_cisco_rules` is the Semgrep control for `ai_deep_sast_quick`, with
    the same detections
  - Vulnhuntr reports no line numbers, so only snippet-level metrics apply
- Vulnhuntr snippet-level counts above match `results/comparison_tables.md`
  (recall 13/20 seeded + 2/12 CVE = 15/32; FP 4/18 secure + 2/12 CVE = 6/30)
  - Its per-category metrics are shown separately (hits/total, from
    `results/comparison_tables.md`):

    | Category                                         | Recall | FP   |
    | :----------------------------------------------- | :----- | :--- |
    | Seeded                                           | 13/20  | -    |
    | CVE                                              | 2/12   | 2/12 |
    | Secure                                           | -      | 4/18 |
    | A01:2025 Broken Access Control                   | 5/7    | 2/6  |
    | A02:2025 Security Misconfiguration               | 2/2    | 0/1  |
    | A04:2025 Cryptographic Failures                  | 0/4    | 1/4  |
    | A05:2025 Injection                               | 5/10   | 2/12 |
    | A07:2025 Authentication Failures                 | 1/5    | 0/4  |
    | A08:2025 Software or Data Integrity Failures     | 2/2    | 1/2  |
    | A09:2025 Security Logging and Alerting Failures | 0/1    | 0/1  |
    | A10:2025 Mishandling of Exceptional Conditions   | 0/1    | -    |

- Observation: `ai_deep_sast_quick`, `semgrep_cisco_rules`, and `vulnhuntr` have
  identical snippet-level counts
  - The first two flag the same snippets: quick mode runs the same Semgrep rules
  - Vulnhuntr matches only by coincidence: its paired accuracy differs (0.24 vs
    0.19), so it flags a different set of snippets
- CVEs caught, of 12: Bandit 2, Semgrep 0, plain LLM 7, Vulnhuntr 2
- Per-category breakdowns are in `results/comparison_tables.md`

- Bandit also flags both fixed versions of the CVEs it catches
- Neither tool finds missing login, IDOR, path traversal, plain-text password
  storage, XXE via `lxml`, or a check that fails open

# Structure of the Dir

| Dir                        | Description                                                         |
| :------------------------- | :------------------------------------------------------------------ |
| `benchmark/seeded/`        | 20 snippets with one planted vulnerability each                      |
| `benchmark/secure/`        | 18 safe snippets that look like vulnerabilities                     |
| `benchmark/cve/`           | 12 real CVEs as `_vuln` and `_fixed` pairs                          |
| `benchmark/exploit_tests/` | Tests that run each attack to prove the labels                      |
| `docker/`                  | Vulnhuntr Docker image and the hook that logs its API calls          |
| `runners/`                 | One script per agent, writing normalized findings                   |
| `results/raw/`             | Anonymized snippets, raw agent output, and the API usage ledger, git-ignored |
| `results/normalized/`      | Findings and run metadata in one format, one file per run           |
| `test/`                    | Unit tests for the libraries                                        |
| `third_party/`             | Checkouts of agents, e.g., Vulnhuntr, git-ignored                   |

# Description of Files

| File                     | Description                                                          | Cluster    |
| :----------------------- | :------------------------------------------------------------------- | :--------- |
| `.env.example`           | Names of the API keys to put in `.env`                               | Setup      |
| `benchmark/labels.csv`   | One row per snippet, generated from the snippet headers              | Benchmark  |
| `benchmark_lib.py`       | Parse headers, validate the benchmark, strip answers, anonymize      | Benchmark  |
| `findings_lib.py`        | Normalized finding format, SARIF and Bandit readers, storage         | Findings   |
| `llm_lib.py`             | Prompt, cached OpenAI-compatible calls, answer parsing               | Agents     |
| `plan.md`                | Step-by-step codebase setup plan                                     | Docs       |
| `rate_limit_lib.py`      | Rate limit headers, daily vs per-minute 429s, shared usage ledger    | Agents     |
| `pytest.ini`             | Loads the course `conftest.py` and sets test paths                   | Setup      |
| `requirements.txt`       | Python packages for the project venv                                 | Setup      |
| `runner_lib.py`          | Shared steps of all runners: anonymize, map back, write the run      | Agents     |
| `score_results.py`       | Score every run and print the metrics table                          | Evaluation |
| `scoring_lib.py`         | Match findings to labels and compute metrics                         | Evaluation |
| `setenv.sh`              | Activate the venv and add the course `helpers` to `PYTHONPATH`       | Setup      |
| `update_labels.py`       | Validate the benchmark and regenerate `labels.csv`                   | Benchmark  |
| `vulnhuntr_lib.py`       | Parse Vulnhuntr logs into findings, summarize its API calls          | Agents     |

# Setup

- Prerequisites:
  - A clone of the course repo next to this repo, i.e., `../umd_classes`
    - The code imports the course `helpers` library from
      `../umd_classes/helpers_root`
  - [`uv`](https://docs.astral.sh/uv/)
  - Docker Desktop, for Vulnhuntr

- Create the venv and install the packages:
  ```bash
  > uv venv --python 3.11 .venv
  > source .venv/bin/activate
  > uv pip install -r requirements.txt
  ```

- Set up every new shell:
  ```bash
  > source setenv.sh
  ```
  - If the course repo is elsewhere, set `HELPERS_ROOT` first:
    ```bash
    > export HELPERS_ROOT=/path/to/umd_classes/helpers_root
    ```

- Add the free API key (only needed for LLM agents):
  ```bash
  > cp .env.example .env
  ```
  - Paste a Groq key after `GROQ_API_KEY=`
  - `.env` is git-ignored: never commit it

- Build the Vulnhuntr image (only needed for Vulnhuntr):
  ```bash
  > git clone https://github.com/protectai/vulnhuntr.git third_party/vulnhuntr
  > git -C third_party/vulnhuntr checkout ead88c5
  > docker build -f docker/vulnhuntr.Dockerfile -t vulnhuntr:ead88c5 \
      third_party/vulnhuntr
  ```
  - The image never uses the project venv
  - The Dockerfile works around 2 upstream packaging bugs (see its header)
  - Check: `docker run --rm vulnhuntr:ead88c5 --help`

- Check that everything works:
  ```bash
  > python -m pytest -q
  > update_labels.py --check
  ```

# Description of Executables

| Command                       | Description                                                  |
| :---------------------------- | :----------------------------------------------------------- |
| `update_labels.py`            | Validate all snippets and regenerate `benchmark/labels.csv`  |
| `runners/run_semgrep.py`      | Run Semgrep CE on the anonymized benchmark                   |
| `runners/run_bandit.py`       | Run Bandit on the anonymized benchmark                       |
| `runners/run_plain_llm.py`    | Ask an LLM with one fixed prompt to review each snippet      |
| `runners/run_ai_deep_sast.py` | Run Cisco deep analysis, or check indexing with `--dry_run` |
| `runners/run_ai_deep_sast_quick.py` | Run Cisco's Semgrep-first scan with local Foundation-Sec-8B |
| `runners/run_vulnhuntr.py`    | Run Vulnhuntr in Docker on each snippet                      |
| `score_results.py`            | Score every run and write `results/scores.csv`               |

## `update_labels.py`

- What it does:
  - Checks every snippet header, bug-line marker, size, and pair
  - Rejects words in the code that leak the label, e.g., `# safe query`
  - Regenerates `labels.csv`, or only checks it with `--check`

- Validate and regenerate the labels after editing snippets:
  ```bash
  > update_labels.py
  ```

- Check that the labels are in sync, e.g., before a commit:
  ```bash
  > update_labels.py --check
  ```

## `runners/run_semgrep.py` and `runners/run_bandit.py`

- What they do:
  - Write anonymized copies of the snippets to `results/raw/<agent>/<run_id>/`
  - Run the tool with default security rules and keep its raw output
  - Write `results/normalized/<agent>/<run_id>.findings.jsonl` and `.meta.json`

- Run both baselines:
  ```bash
  > runners/run_semgrep.py --run_id r1
  > runners/run_bandit.py --run_id r1
  ```

## `runners/run_plain_llm.py`

- What it does:
  - Sends each anonymized snippet, with line numbers, to the model
  - Caches every response, so a run stopped by a rate limit restarts for free
  - Works with any OpenAI-compatible API; the default is the Groq free tier

- Smoke test on 2 snippets:
  ```bash
  > runners/run_plain_llm.py --run_id smoke --max_snippets 2
  ```

- Full run:
  ```bash
  > runners/run_plain_llm.py --run_id r1
  ```

- Use the Gemini free tier instead:
  ```bash
  > runners/run_plain_llm.py \
      --run_id gemini_r1 \
      --agent plain_llm_gemini \
      --model gemini-2.5-flash \
      --base_url https://generativelanguage.googleapis.com/v1beta/openai/ \
      --api_key_env GEMINI_API_KEY
  ```

## `runners/run_vulnhuntr.py`

- What it does:
  - Runs one container per snippet, with only that snippet mounted
  - Passes the key from `.env` to Docker as `OPENAI_API_KEY`, by name only
  - Points Vulnhuntr at Groq: `OPENAI_BASE_URL=https://api.groq.com/openai/v1`,
    `OPENAI_MODEL=openai/gpt-oss-120b`
  - Turns each vulnerability type in Vulnhuntr's final report into one
    finding, with no line, if its confidence is at least `--min_confidence`
    (default 7, Vulnhuntr's own threshold)
  - Keeps each snippet's logs in `results/raw/vulnhuntr/<run_id>/out/`, so a
    rerun skips finished snippets whose code is unchanged

- Smoke test on 1 snippet:
  ```bash
  > runners/run_vulnhuntr.py --run_id smoke --snippet_ids SEED-001
  ```

- Full run:
  ```bash
  > runners/run_vulnhuntr.py --run_id r1
  ```

## Rate Limits

- Both LLM runners:
  - Retry a per-minute 429 after its `retry-after` time
  - Stop with exit code 2 at the daily limit, writing no partial results:
    rerun the same command after the limit resets and the caches skip the
    finished snippets
  - Log the remaining requests (per day) and tokens (per minute) after each
    call
  - Append each call to `results/raw/groq_usage.jsonl`, since no header
    reports the tokens left today
    - A runner does not start a call (plain LLM) or a snippet (Vulnhuntr)
      that would go over `--daily_token_budget` (default 200,000) in the last
      24 hours
- Vulnhuntr cannot resume inside a snippet, so it only starts a snippet when
  `--snippet_token_reserve` (default 20,000) tokens are left
  - Its API calls are logged by `docker/vulnhuntr_hooks/sitecustomize.py`,
    which wraps the HTTP client without changing Vulnhuntr's code
- Check today's usage:
  ```bash
  > python -c "import rate_limit_lib as r; \
      print(r.get_tokens_in_window(r.DEFAULT_LEDGER_FILE))"
  ```

## `score_results.py`

- What it does:
  - Refuses to score if the benchmark is invalid or `labels.csv` is out of sync
  - Scores a partial run (e.g., a smoke test) only on the snippets it reviewed
  - Prints one row of metrics per run and writes `results/scores.csv`

- Score all runs:
  ```bash
  > score_results.py
  ```

# Description of Workflows

- Add or change a snippet:
  - Edit the file in `benchmark/<type>/` and mark each bug line with
    `# {!LINE}`
  - Run `update_labels.py`, then `python -m pytest -q`
- Run an agent and compare it:
  - Run its runner with a new `--run_id`
  - Run `score_results.py`

# Description of Architecture

- Labels:
  - The header docstring of each snippet is the single source of truth
  - `labels.csv` is generated from the headers and checked against them
- Anonymization:
  - Agents see copies without the header or markers
  - Copies have neutral names (`case_<hash>.py`), so `SAFE-` or `_vuln` never
    leak
  - `case_map.csv` maps findings back to snippet ids and original line numbers
- Runs:
  - Every agent writes the same finding format (`findings_lib.FINDING_COLUMNS`)
  - Run metadata records tool version, model, timing, and tokens
- Code style:
  - Follows the course coding conventions
  - Uses the course `helpers` library, so the code moves to `umd_classes`
    without changes