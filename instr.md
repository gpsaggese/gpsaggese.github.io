helpers_root/dev_scripts_helpers/system_tools/create_links.py and stage_links.py

print only the basename of the files

Instead of printing
```

src_file                                                                    | dst_file                                                                                                                      | current_state | target_state |
--------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------ |
class_project/project_template/.dockerignore                                | -                                                                                                                             | missing       | -            |
-                                                                           | msml610/tutorials/L03_knowledge_representation/.ipynb_checkpoints/L03_01_entailment_implication_inference-checkpoint.ipynb    | extra         | -            |
-                                                                           | msml610/tutorials/L03_knowledge_representation/.ipynb_checkpoints/L03_01_entailment_implication_inference_utils-checkpoint.py | extra         | -            |
-                                                                           | msml610/tutorials/L03_knowledge_representation/.ipynb_checkpoints/L03_02_wumpus_world-checkpoint.ipynb                        | extra         | -            |
```

print

```
src_dir=class_project/project_template/
dst_dir=msml610/tutorials/L03_knowledge_representation/

src_file                                                                    | dst_file                                                                                                                      | current_state | target_state |
--------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------ |
.dockerignore                                | -                                                                                                                             | missing       | -            |
-                                                                           | L03_01_entailment_implication_inference-checkpoint.ipynb    | extra         | -            |
-                                                                           | L03_01_entailment_implication_inference_utils-checkpoint.py | extra         | -            |
-                                                                           | L03_02_wumpus_world-checkpoint.ipynb                        | extra         | -            |
```

## Plan
- [x] Read `create_links.py` and `stage_links.py`, and `helpers/htable.py` to
      understand table construction/printing
- [x] `create_links.py`:
  - [x] `_build_status_table()`: report `src_file`/`dst_file` as basenames
        (keep `-` for the missing side)
  - [x] `_build_stage_status_table()`: report `link`/`target_file` as
        basenames
  - [x] `_main()`: print `src_dir=...`/`dst_dir=...` header before the
        `--replace_links` table, and `dst_dir=...` header before the
        `--stage_links` table
  - [x] Update the docstrings of the two `_build_*_table()` functions to
        note the basename-only reporting
- [x] `stage_links.py`:
  - [x] `build_status_table()`: report `link`/`target_file` as basenames
  - [x] `main()`: print `dst_dir=...` header before the table
  - [x] Update the docstring of `build_status_table()`
- [x] Sanity-check both scripts still parse/run (`python3 -c "import ..."` /
      `--help`)
- [x] `git add` the two modified files (no commit)

## Result
- Done: `create_links.py` and `stage_links.py` now print only the basename
  of each file in the table, with a `src_dir=`/`dst_dir=` (or `dst_dir=`
  only, for the stage tables) header printed once above the table
  - Verified with dry-run invocations of `--replace_links`, `--stage_links`,
    and the standalone `stage_links.py` against real dirs in the repo
  - Staged (`git add`, not committed) both modified files inside the
    `helpers_root` submodule
- Not done: nothing outstanding from this task
  - `helpers_root/dev_scripts_helpers/system_tools/test/test_lib_ffind.py`
    and a new `test/outcomes/Test_main.test1/` dir showed up as
    modified/untracked in `git status` inside `helpers_root`; these are
    pre-existing, unrelated to this task, and were left untouched
