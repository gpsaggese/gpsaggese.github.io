In http://localhost:8888/lab/tree/git_root/msml610/tutorials/L03_knowledge_representation/L03_01_entailment_implication_inference.ipynb

1) Always use bullet points in markdown

Do not do this

_Model table_: the same 4-row table, with $M(KB)$ shaded blue and
$M(\alpha)$ outlined in dashed orange
_Inclusion counts_: bar chart of $|M(KB)|$, the overlap with $M(\alpha)$,
and the counterexample rows
_Comments_: which `KB` sentences are toggled on, the query $\alpha$, and
the entailment verdict

but

- _Model table_: the same 4-row table, with $M(KB)$ shaded blue and $M(\alpha)$ outlined in dashed orange
- _Inclusion counts_: bar chart of $|M(KB)|$, the overlap with $M(\alpha)$, and the counterexample rows
- _Comments_: which `KB` sentences are toggled on, the query $\alpha$, and the entailment verdict


2) In cell1_1_models_and_satisfaction, remove model count

3) In cell2_1_entailment_model_checking, remove model inclusion

4) Add a KB to clarify that 
Rain
Rain => WetGround

are the knowledge base

5) Replace

Rain / Rain => WetGround

with 

KB: "Rain", "Rain => WetGround"

## Plan
- Target files (edit the jupytext-paired `.py`, then sync to `.ipynb`):
  - `msml610/tutorials/L03_knowledge_representation/L03_01_entailment_implication_inference.py`
  - `msml610/tutorials/L03_knowledge_representation/L03_01_entailment_implication_inference_utils.py`
- [x] Item 1: bullet-list all non-bulleted `_Label_: ...` description blocks
  in the `.py` markdown cells (each has this pattern; fixing all, not just
  the cited example, since the rule says "Always")
  - [x] Cell 2.1 block (the one quoted in `instr.md`) — also drop the
    `_Inclusion counts_` bullet here since item 3 removes that panel
  - [x] Cell 2.2 block
  - [x] Cell 3.1 block
  - [x] Cell 4.1 block
- [x] Item 2: in `cell1_1_models_and_satisfaction` (utils), remove the
  "Model count" bar-chart panel; go from 3 subplots to 2, keep model table
  + comments panels
- [x] Item 3: in `cell2_1_entailment_model_checking` (utils), remove the
  "Model inclusion" bar-chart panel; go from 3 subplots to 2, keep model
  table + comments panels
  - [x] `draw_count_bars` becomes unused after items 2 and 3 — removed the
    now-dead helper function too
- [x] Items 4-5 (same edit): in `cell2_1_entailment_model_checking`'s
  `param_info`, replace the key `"Rain / Rain => WetGround"` with
  `'KB: "Rain", "Rain => WetGround"'` so the info panel reads the two
  toggles as the `KB`
- [x] Run `jupytext --sync` to regenerate the `.ipynb` from the edited `.py`
  - reverted the incidental `jupytext_version` header bump (local jupytext
    is older than the one that last wrote the file)
- [x] Check line lengths stay within the repo's 81-column limit
- [x] `git add` the two edited files (no commit)

## Result
- Done:
  - Bulleted the 4 non-bulleted `_Label_: ...` description blocks (cells
    2.1, 2.2, 3.1, 4.1) in the paired `.py` markdown cells
  - Removed the "Model count" panel from `cell1_1_models_and_satisfaction`
    and the "Model inclusion" panel from `cell2_1_entailment_model_checking`
    (3-panel layouts → 2-panel; model table + comments panel only)
  - Removed the now-unused `draw_count_bars` helper (no remaining callers)
  - Relabeled the `param_info` key from `"Rain / Rain => WetGround"` to
    `'KB: "Rain", "Rain => WetGround"'` in `cell2_1_entailment_model_checking`
  - Synced `.ipynb` from the edited `.py` via `jupytext --sync`
  - Staged the two edited `.py`/`_utils.py` files with `git add`
- Not done:
  - Did not touch the pre-existing uncommitted diff in the `.ipynb`
    (stale widget execution output on the Cell 2.2 code cell) — it predates
    this task and is unrelated to the 5 items
  - Did not re-run the notebook to refresh outputs — out of scope for a
    source/text edit task
  - Left `docker_build.version.log`, `msml610/tutorials/docker_build.log`,
    and the `tmp.*` files untracked, per the "no temp files" rule
