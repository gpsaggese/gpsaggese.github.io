### [ ] Standardize executable CLI interfaces

* Repo:
- umd_classes
- helpers

* Execution type: GitHub

* Problem
- Executables across the repo use inconsistent flags for the same concept (input
  file and file list), making scripts harder to chain and remember

* Implementation Notes
- Scope: audit these directories in both repos
  - umd_classes: `dev_scripts_umd_classes/`
  - helpers: `helpers_root/dev_scripts_helpers/`, `helpers_root/linters2/`
- Invoke targets: check tasks in `tasks.py`, `msml610/tasks.py`,
  `helpers_root/tasks.py`
- Standard interface: `-i`/`--input` for single file, `-f`/`--files` for list
- Shared utilities: place in `helpers_root/`, import from both repos
- Batch strategy: one directory of scripts per PR (max ~3-5 scripts)

* Solution

- [x] Audit and document the standard interface

  **Plan:**
  1. Scan all executables in `dev_scripts_umd_classes/`, `helpers_root/dev_scripts_helpers/`,
     `helpers_root/linters2/` for argparse flags using grep/ast parsing
  2. Audit invoke targets in `tasks.py`, `msml610/tasks.py`, `helpers_root/tasks.py` for
     parameter inconsistency
  3. Create `audit_results.md` with raw findings: script name, current flags, deviation type
  4. Document standard: `-i`/`--input` (single), `-f`/`--files` (list), add to `plan.md`
  5. Create `plan.md` with standardization strategy (e.g., shared utility in `helpers_root/`,
     batch-by-directory rollout)

- [-] Fix deviating scripts

  **Plan:**
  1. Create shared utility function in `helpers_root/helpers/hparser.py` for standardized arg handling ✓
  2. Update Batch 1: `dev_scripts_umd_classes/lesson_parser.py` (--input_file → -i/--input)
  3. Test locally and in CI
  4. Update remaining batches (2-8) per PR strategy

  **Progress:**
  - [x] Added `add_input_file_arg()`, `add_files_list_arg()`, `add_input_dir_arg()` to hparser.py
  - [x] Tested new functions locally (all pass)
  - [x] Fix Batch 1 script: lesson_parser.py (--input_file → -i/--input)
  - [ ] Fix remaining batches (2-8)
    - Batch 2: coding_tools/ (5 scripts)
    - Batch 3: documentation/ (4 scripts)
    - Batch 4: github/ (5 scripts)
    - Batch 5: notebooks/ (3 scripts)
    - Batch 6: llms/ (1 script)
    - Batch 7: input_dir normalization (5+ scripts)
    - Batch 8: invoke tasks (if needed)

