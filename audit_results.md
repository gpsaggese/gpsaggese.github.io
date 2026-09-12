# Audit: CLI Interface Flags for Input/Files

## Summary

- **Total scripts audited**: 90+ Python executables across `dev_scripts_umd_classes/`, `helpers_root/dev_scripts_helpers/`, `helpers_root/linters2/`
- **Standard compliant** (`-i`/`--input` for single file, `-f`/`--files` for lists): ~40 scripts
- **Deviating** (using `--input_file`, `--file`, `--in_file`, etc.): ~35 scripts
- **Unrelated flags**: remainder (these don't use file input args)

---

## Standard Interface Definition

### Single Input File
- **Flags**: `-i` / `--input`
- **Aliases**: `--input_file` (deprecated, migrate to `-i`/`--input`)
- **Type**: string (file path or `-` for stdin)
- **Example**: `script.py -i data.txt` or `script.py --input data.txt`

### File List
- **Flags**: `-f` / `--files`
- **Type**: string or comma-separated list
- **Example**: `script.py -f file1.txt,file2.txt` or `script.py -f file1.txt file2.txt`

### Input Directory
- **Flags**: `--input_dir`
- **Type**: string (directory path)
- **Used by**: scripts that process entire directories
- **Example**: `compress_figures.py --input_dir ./images/`

---

## Scripts Currently Following Standard

### Single Input (`-i`/`--input`)
✅ Compliant (no action needed):
- `traceback_to_cfile.py`
- All `dockerized_*` scripts (8 scripts in `dockerize/`)
- `clean_markdown.py`
- `convert_pdf_to_md.py`
- `md_to_speech.py`
- `notes_to_pdf.py`
- `preprocess_notes.py`
- `count_words.py` (partially: has both `-i` and `--input_file`)
- `pytest_failed.py`
- `to_github.py`
- And ~15 others in documentation/download subdirectories

### File Lists (`-f`/`--files`)
✅ Compliant (no action needed):
- `gd_notebook.py` (-f / --files)
- `old/linter/linter.py` (-f / --files)
- `generate_videos/convert_png_to_movie.py` (--files)
- `copy_across_clients.py` (--files)

---

## Scripts Deviating from Standard

### Deviation Type 1: `--input_file` instead of `-i`/`--input`
🔴 Needs migration:
- `lesson_parser.py` → migrate to `-i`/`--input`
- `reorder_python_code.py` → migrate to `-i`/`--input`
- `split_in_files.py` → migrate to `-i`/`--input`
- `github/sync_gh_issue_labels.py` → migrate to `-i`/`--input`
- `github/dockerized_sync_gh_issue_labels.py` → migrate to `-i`/`--input`
- `count_words.py` (conflicting: has `-i` AND `--input_file`) → standardize to `-i`/`--input`
- `github/dockerized_sync_gh_repo_settings.py` → migrate to `-i`/`--input`

### Deviation Type 2: `--file` or `--in_file` instead of `-i`/`--input`
🔴 Needs migration:
- `replace_latex.py` (--file) → migrate to `-i`/`--input`
- `github/gh_migration/bulk_transfer_issues.py` (--file) → migrate to `-i`/`--input`
- `notebooks/publish_notebook.py` (--file) → migrate to `-i`/`--input`
- `github/set_secrets_and_variables.py` (--file) → migrate to `-i`/`--input`
- `toml_merge.py` (--in_file) → migrate to `-i`/`--input`
- `check_links.py` (--in_file) → migrate to `-i`/`--input`
- `transform_pandoc_ast_to_typst.py` (--in_file) → migrate to `-i`/`--input`
- `documentations/extract_from_md.py` (docs mention but not verified) → migrate to `-i`/`--input`
- `llms/dockerized_llm_review.py` (-i / --in_file_path) → standardize to `-i`/`--input`

### Deviation Type 3: `--file_name` or similar naming variants
🔴 Needs migration:
- `process_prof.py` (--file_name) → migrate to `-i`/`--input`
- `google/to_local_dir.py` (--file_name) → use case specific, needs review
- `notebooks/extract_notebook_images.py` (--in_notebook_filename) → use case specific, needs review

### Deviation Type 4: `--from_file` instead of `-i`/`--input`
🔴 Needs migration:
- `diff_to_vimdiff.py` (--from_file) → migrate to `-i`/`--input`

### Deviation Type 5: `--input_files` instead of `-f`/`--files`
🔴 Needs migration:
- `concatenate_pdfs.py` (--input_files) → migrate to `-f`/`--files`
- `notebooks/add_toc_to_notebook.py` (--input_files) → migrate to `-f`/`--files`

### Deviation Type 6: Directory input variants (need separate standard)
⚠️ Review/normalize:
- `compress_figures.py` (--input_dir)
- `extract_gdoc_map.py` (--input_dir)
- `convert_png_dir_to_movie.py` (--input_dir)
- `notebooks/add_toc_to_notebook.py` (--input_dir) — also has --input_files
- `generate_videos/create_presentation_video.py` (--in_dir)
- `generate_videos/generate_synthesia_videos.py` (--in_dir)
- `generate_videos/extract_png_from_ppt.py` (--in_file)
- `documentation/encrypt_models/encrypt_model.py` (--input_dir)

**Recommendation**: Standardize to `--input_dir` for directories

---

## Invoke Tasks (`.py` files with `@task` decorator)

Need to audit:
- `helpers_root/tasks.py` — check git-related tasks (git_branch_diff, etc.)
- `msml610/tasks.py` — check course-specific tasks
- `tasks.py` (root) — check repo-wide tasks

**Common patterns observed**: Many invoke tasks use custom `--input`, `-f`, `--files` inconsistently.

---

## Standardization Strategy

### Phase 1: Create shared utility (helpers_root/)
Create `helpers_root/lib_cli_utils.py` with:
- `add_input_file_arg()` — standardizes `-i`/`--input` across all scripts
- `add_files_list_arg()` — standardizes `-f`/`--files` across all scripts
- `add_input_dir_arg()` — standardizes `--input_dir` across all scripts

### Phase 2: Migrate by directory (batch PRs)

**Batch 1: UMD Classes root**
- `dev_scripts_umd_classes/lesson_parser.py`

**Batch 2: coding_tools/**
- `reorder_python_code.py`
- `split_in_files.py`
- `toml_merge.py`
- `diff_to_vimdiff.py`
- `process_prof.py`

**Batch 3: documentation/**
- `replace_latex.py`
- `check_links.py`
- `transform_pandoc_ast_to_typst.py`
- `count_words.py` (fix mixed flags)

**Batch 4: github/**
- `sync_gh_issue_labels.py`
- `dockerized_sync_gh_issue_labels.py`
- `dockerized_sync_gh_repo_settings.py`
- `gh_migration/bulk_transfer_issues.py`
- `set_secrets_and_variables.py`

**Batch 5: notebooks/**
- `publish_notebook.py`
- `add_toc_to_notebook.py`
- `extract_notebook_images.py`

**Batch 6: llms/**
- `dockerized_llm_review.py`

**Batch 7: Directory inputs** (--input_dir normalization)
- All scripts using `--input_dir`, `--in_dir` → standardize to `--input_dir`

**Batch 8: Invoke tasks**
- Audit and fix `tasks.py` files (if they use argparse internally)

---

## Notes

- Some scripts use alternative input methods (`--from_pb`, `--from_latest_file`, `--from_scratch`) — these are OKAY, they're additional options
- Mutually exclusive groups (e.g., `traceback_to_cfile.py`) are acceptable
- Short flags like `-i`, `-f` and long flags `--input`, `--files` should always coexist (both forms)
- Some scripts take custom arguments by design (e.g., `--output`, `-o` for output) — keep those unchanged
