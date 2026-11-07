<!-- toc -->

- [Summary](#summary)
- [Scope](#scope)
- [Commands](#commands)
- [Approval Gates](#approval-gates)
- [Prepare the Environment](#prepare-the-environment)
- [Run the Offline Demo](#run-the-offline-demo)
- [Validate the Live Plan](#validate-the-live-plan)
- [Run the Paid Warm-Up](#run-the-paid-warm-up)
- [Estimate Cost](#estimate-cost)
- [Stop the Run](#stop-the-run)
- [Store the Evidence](#store-the-evidence)
- [Present the Demo](#present-the-demo)
- [Troubleshoot](#troubleshoot)
- [Rehearsal Record](#rehearsal-record)
- [Completion Checklist](#completion-checklist)

<!-- tocstop -->

# Run the Checker Demo and Live Test

## Summary

- This guide prepares the Noesis checker demo and the first paid provider test
- The workflow has two separate goals:
  - Reproduce the offline checker validation from PR #832
  - Warm up the live monitoring pipeline with a bounded paid sample
- The workflow proceeds in stages:
  - Offline replay
  - Validation-only live plan
  - Five-call paid smoke test
  - Ten-call paid warm-up
- The operator must not make paid calls before GP approves:
  - Provider and exact model
  - Budget
  - Stop conditions
- The paid stages also depend on the standalone monitoring runtime in PR #739

## Scope

- This guide supports:
  - Issue #840: Checker demo and paid live-test plan
  - Issue #684: Quality-of-service paper and technical report
  - Issue #645: Future live degradation detector
- The offline replay establishes that:
  - Checker verdicts match the analytic predictions
  - Latency and quality failures remain separately attributable
  - Missing evidence does not create a false pass or failure
- The live warm-up establishes that:
  - A pinned provider and model can be probed
  - Responses can be graded and persisted
  - Latency, tokens, cost, status, and model attribution can be recorded
- This guide does not:
  - Modify `NoesisMarket`
  - Implement the longitudinal degradation detector
  - Prove that a live model was substituted or nerfed
  - Authorize provider spending

## Commands

| Command | Purpose | Provider Calls | Database Writes |
| :------ | :------ | :------------- | :-------------- |
| `run_checker_replay.py` | Reproduce the offline validation | 0 | 0 |
| `pytest` focused suite | Check replay logic and CLI | 0 | 0 |
| `generate_canary_bank.py` | Generate candidate canaries | 0 | 0 |
| `monitor_cli` without `--execute` | Validate a live plan | 0 | 0 |
| `monitor_cli --execute` | Run an approved paid sample | 5 or 10 | Yes |

## Approval Gates

- Complete every gate before adding `--execute`:
  - [ ] GP has reviewed the live-test plan
  - [ ] PR #739 or an approved replacement is merged
  - [ ] The repository revision is recorded
  - [ ] `helpers_root` is initialized at the pinned revision
  - [ ] The provider and exact model identifier are approved
  - [ ] Provider fallback and routing behavior are disabled or recorded
  - [ ] Candidate canaries are calibrated, reviewed, and active
  - [ ] The canary-bank version and file hashes are recorded
  - [ ] The grader version and request settings are recorded
  - [ ] PostgreSQL is reachable and migrations are current
  - [ ] Current provider prices and the worst-case cost are recorded
  - [ ] GP approves the dollar cap and stop conditions
  - [ ] One operator is responsible for the run and manual stop
- Store credentials only in the local secret environment
  - Do not put credentials in Git, commands, reports, screenshots, or the
    research Google Doc

## Prepare the Environment

- Start from a clean repository checkout:
  ```bash
  > git status --short
  > git rev-parse HEAD
  > git submodule status
  > python3 --version
  > python3 -m pytest --version
  ```
- Record the outputs in the run manifest
- Configure live-run secrets outside the repository:
  - `OPENROUTER_API_KEY`
  - `NOESIS_CANARY_SCHEDULE_SECRET`
  - `NOESIS_DATABASE_URL`
- Record the non-secret policy settings:
  - Provider slug
  - Requested model identifier
  - Maximum output tokens
  - Per-request timeout
  - Experiment spend cap

## Run the Offline Demo

- Run the full deterministic replay from the repository root:
  ```bash
  > research/Noesis/scripts/run_checker_replay.py \
      --output_dir /tmp/noesis_checker_replay
  ```
- Expect the command to:
  - Use five seeds
  - Exercise six scenarios
  - Simulate 30,000 contracts
  - Print `Overall acceptance: PASS`
  - Make no provider calls or database writes
- Inspect the generated evidence:
  - `/tmp/noesis_checker_replay/checker_replay.json`
  - `/tmp/noesis_checker_replay/checker_replay.csv`
  - `/tmp/noesis_checker_replay/checker_replay.md`
- Run a shorter meeting demo without replacing checked-in findings:
  ```bash
  > research/Noesis/scripts/run_checker_replay.py \
      --seeds 0,1 \
      --contracts_per_seed 100 \
      --output_dir /tmp/noesis_checker_replay_fast
  ```
- Run the focused tests:
  ```bash
  > pytest -q \
      research/Noesis/checker/test/test_checker_replay.py \
      research/Noesis/checker/test/test_checker_replay_runner.py \
      research/Noesis/checker/test/test_checker_replay_report.py \
      research/Noesis/scripts/test/test_run_checker_replay.py
  ```
- Check byte-for-byte reproducibility:
  ```bash
  > research/Noesis/scripts/run_checker_replay.py \
      --output_dir /tmp/noesis_checker_replay_1
  > research/Noesis/scripts/run_checker_replay.py \
      --output_dir /tmp/noesis_checker_replay_2
  > diff -r \
      /tmp/noesis_checker_replay_1 \
      /tmp/noesis_checker_replay_2
  ```
- Expect `diff` to print no output and return exit status zero

## Validate the Live Plan

- Generate the deterministic candidate bank in a disposable directory:
  ```bash
  > python3 research/Noesis/scripts/generate_canary_bank.py \
      --output-dir /tmp/noesis_candidate_bank
  > shasum -a 256 /tmp/noesis_candidate_bank/*.jsonl
  ```
- Expect 300 candidate canaries across six families
- Treat generated questions as candidates, not trusted quality evidence
  - Live calibration decides which questions can become active canaries
- After the standalone runtime is merged, validate a ten-call plan:
  ```bash
  > python3 -m research.Noesis.monitor.monitor_cli \
      --target PROVIDER_SLUG=EXACT_MODEL_ID \
      --bank-dir /tmp/noesis_candidate_bank \
      --canary-count 10
  ```
- Do not add `--execute` during plan validation
- Expect output similar to:
  ```text
  Monitoring plan: 10 canaries x 1 target = 10 paid provider calls
  Validation only: no database writes or provider calls were made.
  ```

## Run the Paid Warm-Up

- Run paid stages only after every approval gate passes
- Stage 1: Five-call smoke test
  - Use one provider and one exact model
  - Use five active calibrated canaries
  - Use temperature zero
  - Limit output to 64 tokens
  - Use a 30-second per-request timeout
  - Disable automatic retries
  - Apply the approved dollar cap
  ```bash
  > python3 -m research.Noesis.monitor.monitor_cli \
      --target PROVIDER_SLUG=EXACT_MODEL_ID \
      --canary-count 5 \
      --max-tokens 64 \
      --timeout 30 \
      --execute
  ```
- Validate Stage 1 before continuing:
  - [ ] No more than five calls were attempted
  - [ ] Every observation has the expected provider and model attribution
  - [ ] Every observation has a timestamp, latency, grade, and status
  - [ ] The database run reached a terminal state
  - [ ] Actual tokens and cost were recorded
  - [ ] Provider and grader failures remained visible
  - [ ] No credentials appeared in the evidence
- Stage 2: Ten-call warm-up
  - Continue only if Stage 1 satisfies every validation item
  - Change only the canary count from five to ten
  - Keep the provider, model, grader, routing, token limit, and timeout frozen
  - Do not scale beyond ten calls under issue #840

## Estimate Cost

- Calculate the estimate immediately before requesting budget:
  ```text
  estimated_cost = requests * (
      estimated_input_tokens_per_request * input_price_per_token
      + max_output_tokens * output_price_per_token
  )
  ```
- Record:
  - Provider pricing URL and access time
  - Input and output prices
  - Expected prompt tokens
  - Maximum output tokens
  - Request count
  - Expected and worst-case cost
- Initial warm-up budget request:
  - Five smoke-test calls
  - Ten warm-up calls
  - 15 calls total
  - 64 maximum output tokens per call
  - 960 maximum generated tokens across both stages
  - `$0.25` provisional hard cap
- Treat `$0.25` as a proposed ceiling, not permission to spend

## Stop the Run

- Stop immediately and do not scale if:
  - Provider or model attribution differs from the frozen plan
  - Routing or fallback cannot be verified
  - Attempted requests exceed the approved count
  - Expected or observed spend would exceed the approved cap
  - Two requests time out
  - Two consecutive provider errors occur
  - Provider or request errors exceed 20 percent
  - Any grader crashes
  - Any observation cannot be persisted
  - Token or cost metadata required for accounting is missing
  - The run cannot be reconciled to one database run identifier
  - The operator cannot confirm that execution stopped
- The PR #739 runtime is sequential and has no retry loop
  - Revisit these limits if concurrency or retries are added

## Store the Evidence

- Assign one immutable run identifier:
  - E.g., `2026-11-06T180000Z_provider_model`
- Store one artifact directory per run:
  ```text
  artifacts/noesis/qos/<run_id>/
  |-- manifest.md
  |-- config.json
  |-- bank_sha256.txt
  |-- console.log
  |-- observations.csv
  |-- summary.json
  |-- summary.md
  `-- plots/
  ```
- Include in `manifest.md`:
  - Issue and PR links
  - Repository and helpers revisions
  - Provider and exact model
  - Request and grader configuration
  - Canary-bank hashes
  - Provider-pricing source
  - Database run identifier
  - Research Google Doc entry
- Use the Google Doc as the human experiment journal
  - Link machine-readable artifacts instead of copying large outputs
- Known export gap:
  - PR #739 stores observations in PostgreSQL and prints a summary
  - PR #739 does not export the full artifact tree or generate plots
  - Define and rehearse a read-only export before paid execution

## Present the Demo

- Use this ten-minute agenda:
  1. State the research question and explicit non-claims
  2. Explain the checker and monitoring evidence flow
  3. Run the fast deterministic replay
  4. Inspect the JSON, CSV, and Markdown findings
  5. Show failure attribution and insufficient evidence
  6. Show the focused tests and reproducibility check
  7. Validate the live plan without `--execute`
  8. Review the budget, stop conditions, and evidence layout
- Request one decision from GP:
  - Approve or revise the provider, exact model, five-call smoke test,
    `$0.25` cap, and criteria for proceeding to ten calls

## Troubleshoot

- `ModuleNotFoundError: No module named 'helpers'`:
  - The `helpers_root` checkout is incomplete
  - Confirm `git submodule status helpers_root`
  - Do not interpret the failure as an experiment result
- Every focused test fails during teardown with `_GLOBAL_CAPSYS` set to `None`:
  - The helper module is importable, but shared `conftest.py` was not loaded
  - Repair the repository setup instead of changing checker code
- `monitor_cli` is not available:
  - PR #739 or its approved replacement is not merged
  - Complete only the offline and validation documentation stages
- No calibrated canaries are active:
  - Do not run paid monitoring
  - Complete calibration, review, and activation first
- A run remains `RUNNING` after a process failure:
  - Stop new executions
  - Preserve logs and the database run identifier
  - Mark or recover the run through an approved database procedure
  - Heartbeats and stale-run cleanup remain future runtime work

## Rehearsal Record

- Rehearsal date: 2026-11-06
- Upstream revision: `cadb6a9319b0223b4c15cf4ce6ec8efcb22dded4`
- Operations performed:
  - Ran the full 30,000-contract replay
  - Ran the focused replay test selection
  - Made no provider calls
  - Made no database writes
  - Spent no money
- Results:
  - Offline replay: `PASS` across all six scenarios
  - Focused tests: 26 passed
  - Replay duration: approximately seven seconds on the rehearsal machine
- Clean-checkout blocker:
  - Current master pins `helpers_root` to `347c111fe110e66ccf462625703192d6e5a779b9`
  - The configured helpers remote did not contain that revision during the
    rehearsal
  - The rehearsal temporarily used an existing local helpers checkout
  - Resolve the pinned revision before claiming clean-checkout reproducibility

## Completion Checklist

- Completed planning and free rehearsal work:
  - [x] Define the demo objective, stages, and expected evidence
  - [x] Record the offline demo and focused-test commands
  - [x] Define the paid warm-up sequence
  - [x] Define the proposed budget and stop conditions
  - [x] Define the artifact layout and demo agenda
  - [x] Rehearse the 30,000-contract replay
  - [x] Rehearse the focused tests
- Complete before paid execution:
  - [ ] Resolve clean `helpers_root` initialization
  - [ ] Merge PR #739 or an approved replacement
  - [ ] Calibrate, review, and activate the intended canaries
  - [ ] Define and rehearse the database artifact export
  - [ ] Freeze the provider, model, grader, bank, and request settings
  - [ ] Record current pricing and exact cost estimate
  - [ ] Obtain GP approval for the budget and stop conditions
- Complete after approval:
  - [ ] Run the five-call smoke test
  - [ ] Validate and attach the evidence
  - [ ] Run the ten-call warm-up if the smoke test passes
  - [ ] Present the result and record the scale or stop decision
