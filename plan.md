# Plan: Standardize Executable CLI Interfaces

## Objective

Unify CLI argument flags across all executables in the repository so developers can reliably chain scripts and remember argument syntax. Standard: `-i`/`--input` for single files, `-f`/`--files` for file lists, `--input_dir` for directories.

---

## Implementation Strategy

### Step 1: Add Shared Utility Functions to `helpers/hparser.py`

**File**: `helpers_root/helpers/hparser.py` (existing module; already has
`add_bool_arg()`, `add_verbosity_arg()` following this exact pattern — add
alongside them rather than creating a new file)

**Functions**:
```python
def add_input_file_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add standardized -i/--input for single file."""
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        type=str,
        required=True,
        help="Input file (or '-' for stdin)"
    )
    return parser

def add_files_list_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add standardized -f/--files for file list."""
    parser.add_argument(
        "-f",
        "--files",
        dest="files",
        type=str,
        nargs="+",
        required=True,
        help="List of input files (space-separated or comma-separated)"
    )
    return parser

def add_input_dir_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add standardized --input_dir for directory input."""
    parser.add_argument(
        "--input_dir",
        dest="input_dir",
        type=str,
        required=True,
        help="Input directory path"
    )
    return parser
```

**Benefits**:
- Single source of truth for CLI interface
- Ensures consistency across all scripts
- Easy to update standards in future

---

### Step 2: Update Scripts in Batches (1 PR per batch)

**Batch Strategy**: One directory per PR to keep review tractable (max 3-5 scripts per PR)

#### Batch 1: UMD Classes root scripts
**PR**: `UmdTask_StandardizeCliInterfaces_Batch1_umd_classes`
- Files to update:
  - `dev_scripts_umd_classes/lesson_parser.py`: `--input_file` → `-i`/`--input`
- Test coverage: check `lesson_parser.py` unit tests if any

#### Batch 2: coding_tools/ directory
**PR**: `UmdTask_StandardizeCliInterfaces_Batch2_coding_tools`
- Files to update:
  - `helpers_root/dev_scripts_helpers/coding_tools/reorder_python_code.py`
  - `helpers_root/dev_scripts_helpers/coding_tools/split_in_files.py`
  - `helpers_root/dev_scripts_helpers/coding_tools/toml_merge.py`
  - `helpers_root/dev_scripts_helpers/coding_tools/diff_to_vimdiff.py`
  - `helpers_root/dev_scripts_helpers/coding_tools/process_prof.py`
- Note: Some scripts may be in `helpers_root/`, test via `helpers_root/` commands

#### Batch 3: documentation/ directory
**PR**: `UmdTask_StandardizeCliInterfaces_Batch3_documentation`
- Files to update:
  - `helpers_root/dev_scripts_helpers/documentation/replace_latex.py`
  - `helpers_root/dev_scripts_helpers/documentation/check_links.py`
  - `helpers_root/dev_scripts_helpers/documentation/transform_pandoc_ast_to_typst.py`
  - `helpers_root/dev_scripts_helpers/documentation/count_words.py`

#### Batch 4: github/ directory
**PR**: `UmdTask_StandardizeCliInterfaces_Batch4_github`
- Files to update:
  - `helpers_root/dev_scripts_helpers/github/sync_gh_issue_labels.py`
  - `helpers_root/dev_scripts_helpers/github/dockerized_sync_gh_issue_labels.py`
  - `helpers_root/dev_scripts_helpers/github/dockerized_sync_gh_repo_settings.py`
  - `helpers_root/dev_scripts_helpers/github/gh_migration/bulk_transfer_issues.py`
  - `helpers_root/dev_scripts_helpers/github/set_secrets_and_variables.py`

#### Batch 5: notebooks/ directory
**PR**: `UmdTask_StandardizeCliInterfaces_Batch5_notebooks`
- Files to update:
  - `helpers_root/dev_scripts_helpers/notebooks/publish_notebook.py`
  - `helpers_root/dev_scripts_helpers/notebooks/add_toc_to_notebook.py`
  - `helpers_root/dev_scripts_helpers/notebooks/extract_notebook_images.py`

#### Batch 6: llms/ directory
**PR**: `UmdTask_StandardizeCliInterfaces_Batch6_llms`
- Files to update:
  - `helpers_root/dev_scripts_helpers/llms/dockerized_llm_review.py`

#### Batch 7: Directory inputs normalization
**PR**: `UmdTask_StandardizeCliInterfaces_Batch7_input_dir`
- Files to update:
  - All scripts using `--input_dir` or `--in_dir` → standardize to `--input_dir`
  - Examples:
    - `helpers_root/dev_scripts_helpers/documentation/compress_figures.py`
    - `helpers_root/dev_scripts_helpers/documentation/extract_gdoc_map.py`
    - `helpers_root/dev_scripts_helpers/generate_videos/`

#### Batch 8: Invoke tasks (if applicable)
**PR**: `UmdTask_StandardizeCliInterfaces_Batch8_invoke_tasks`
- Files to audit:
  - `helpers_root/tasks.py` (git-related tasks)
  - `msml610/tasks.py`
  - `tasks.py` (root)
- Action: Check if invoke tasks use argparse internally; if so, standardize

---

## Steps for Each Batch

### For Each Script:

1. **Update argparse definition**
   ```python
   # OLD
   parser.add_argument("--input_file", type=str, required=True)
   
   # NEW
   from helpers.hparser import add_input_file_arg
   parser = add_input_file_arg(parser)
   ```

2. **Update all call sites** within the same script
   - Replace `args.input_file` → `args.input`
   - Replace `args.file` → `args.input`
   - etc.

3. **Update unit tests**
   - Grep for test invocations: `script.py --input_file` → `script.py -i` or `script.py --input`
   - Update test assertions referencing old flag names

4. **Test locally**
   ```bash
   # Run unit tests for affected modules
   pytest helpers_root/test/test_<script_name>.py -v
   ```

5. **Create PR**
   - Branch: `UmdTask_StandardizeCliInterfaces_Batch<N>_<directory>`
   - Description: List which scripts were updated, summary of changes

---

## Rollback / Deprecation Strategy

### Option 1: Immediate cutover (recommended for small script count)
- Remove old flags entirely
- Pros: Clean, no confusion
- Cons: Breaking change if scripts are called externally

### Option 2: Deprecation period (for widely-used scripts)
- Keep old flags working but print warning: "Flag `--input_file` is deprecated, use `-i` or `--input` instead"
- Set sunset date (e.g., 6 months)
- Remove in final batch

---

## Success Criteria

- [ ] All scripts in `dev_scripts_umd_classes/`, `helpers_root/dev_scripts_helpers/`, `helpers_root/linters2/` use standard flags
- [ ] All unit tests updated and passing
- [ ] No external documentation references old flags
- [ ] Shared utility `lib_cli_utils.py` used in 80%+ of scripts
- [ ] Developers can confidently use `-i`/`--input` and `-f`/`--files` everywhere

---

## Timeline

- **Week 1**: Create `lib_cli_utils.py` + implement Batch 1–2
- **Week 2**: Implement Batch 3–4
- **Week 3**: Implement Batch 5–6
- **Week 4**: Implement Batch 7–8 + final review

---

## Notes

- Some scripts have legitimate alternative input methods (e.g., `--from_pb` for clipboard) — these are **kept** as additional options
- Mutually exclusive argument groups are fine (e.g., `-i` vs `--from_latest_file`)
- Output flags (`-o`, `--output`) are unchanged; this initiative focuses only on **input** arguments
- If a script has special file-handling logic (e.g., glob patterns), document in help text

---

## References

- Audit results: `audit_results.md`
- Shared utility location: `helpers_root/lib_cli_utils.py` (to be created)
- Test framework: pytest with `helpers_root/CLAUDE.md` conventions
