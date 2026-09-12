In http://localhost:8888/lab/tree/git_root/msml610/tutorials/L03_knowledge_representation

1) Add a cell explaining the wumpus world problem in markdown
Explain the breeze axiom 

2) Explain what is the correct strategy to win

3) Shows agent knows only what's told, not hidden truth.

- TELL percept button calls kb.tell_percept(cell), which adds 3 sentences to KB:
  a. Not(P_cell) — agent stood here and survived, so no pit.
  b. Breeze axiom for cell (biconditional linking breeze to neighbor pits).
  c. Observed breeze literal (true/false) from world.percept(cell).

4) In cell2_1_models_and_axiom 

- remove model count
- remove log2(models)

## Plan

- Confirmed with user:
  - Cell 2.1 becomes fully static: `n` fixed at 3, no widget
  - "Remove model count" = remove the bar chart panel only; keep the
    numeric counts in the Comments text panel
  - Tasks 1 and 2 become two separate new markdown cells

- [x] Task 1: add a new markdown cell in `L03_02_wumpus_world.py` (paired
      with the `.ipynb`), placed right after `# Part 1: ...` and before
      `## Cell 1.1`, explaining the wumpus world problem
  - [x] Hidden 4x4 grid, agent starts at `(1, 1)`, pits and the wumpus kill
        the agent, gold is the goal, percepts are breeze / stench / glitter
  - [x] Explain the breeze axiom informally: a breeze is felt in a cell if
        and only if at least one neighboring cell holds a pit
- [x] Task 2: add a second new markdown cell (right after the Task 1 cell,
      still before `## Cell 1.1`) explaining the correct strategy to win
  - [x] Only move into cells the `KB` proves safe, use `TELL`/`ASK` to grow
        certainty, grab the gold once found, return to the start cell
- [x] Task 3: fix the "Key observations" markdown cell after `## Cell 1.1`
      (currently says "every TELL adds one sentence", which is inaccurate)
      to correctly describe that `TELL percept` adds up to three sentences
      to the `KB`
  - [x] `Not(P_cell)`: the agent stood in the cell and survived, so no pit
  - [x] The breeze axiom for the cell, linking the breeze percept to
        neighboring pits
  - [x] The observed breeze literal (true/false) from `world.percept(cell)`
  - [x] Keep the point that the agent only knows what has been `TELL`-ed,
        never the hidden truth
- [x] Task 4: in `cell2_1_models_and_axiom` (`L03_02_wumpus_world_utils.py`)
  - [x] Remove the `log2(models)` slider control, fixing the number of
        enumerated pit variables to 3 (the breeze axiom's own variables),
        making the cell a static (non-interactive) display
  - [x] Remove the "Model count" bar chart panel (`draw_count_bars` call),
        changing the layout from 1x3 to 1x2 subplots (model table +
        comments); keep the numeric counts in the Comments text panel
  - [x] Update the Cell 2.1 markdown (Goal bullets, Key observations) and
        the function docstring to match the simplified, non-interactive
        cell
- [x] Apply matching edits to the paired notebook `L03_02_wumpus_world.ipynb`
      (via jupytext sync)
- [x] Run the notebook top to bottom to confirm it executes without error

## Result

- Done:
  - Added two new markdown cells before `## Cell 1.1` in
    `L03_02_wumpus_world.py`/`.ipynb`: "The Wumpus World Problem" (grid,
    percepts, informal breeze axiom) and "The Correct Strategy to Win"
    (KB-proved safety, no guessing, grab gold, return to start)
  - Fixed the Cell 1.1 "Key observations" markdown, which incorrectly said
    "every TELL adds one sentence"; it now lists the 3 sentences
    `TELL percept` adds (`Not(P_cell)`, the breeze axiom, the observed
    breeze literal) and restates that the agent never sees the hidden truth
  - Simplified `cell2_1_models_and_axiom` in `L03_02_wumpus_world_utils.py`:
    removed the `log2(models)` slider (n fixed at 3, the breeze axiom's own
    variable count) and the "Model count" bar chart; layout is now 1x2
    (model table + comments), comments still report the model counts as
    text
  - Updated Cell 2.1's Goal/Key-observations markdown to match the
    simplified, non-interactive cell
  - Synced the `.ipynb` from the `.py` via `jupytext --sync`, then ran the
    whole notebook inside Docker
    (`docker_cmd.sh "python .../L03_02_wumpus_world.py"`): executed
    top to bottom with no errors
  - Ran `ruff check` on both changed Python files: no new lint issues
    (pre-existing `E402` notebook-import warnings only)
- Not done / flagged:
  - The `.ipynb`'s stored cell outputs (images) were not regenerated; only
    the source cells were synced from the `.py`. Re-run the notebook in
    Jupyter Lab to refresh the displayed plots, especially Cell 2.1's now
    2-panel output
  - `jupytext --sync` also touched
    `L03_03_rule_based_expert_systems.py`/`.ipynb` (only a
    `jupytext_version` metadata bump, `1.19.0` -> `1.19.5`, no content
    change) even though this task never opened that notebook; likely the
    live Jupyter Lab server (see the URL at the top of this file)
    autosaved it during this session. Flagging rather than reverting,
    since it may be a legitimate autosave from your open tab
  - The `helpers_root` submodule now shows as modified too (staged edit to
    `.claude/skills/notebook.rules.md`, plus untracked
    `dev_scripts_helpers/coding_tools/notify.py.log` and
    `helpers/hselect_input_output.py.log`), none of which this task
    touched. It was clean at the start of this session, so something else
    (the live Jupyter Lab server, or another process) changed it
    concurrently. Flagging, not reverting
  - No files were staged (`git add`): all edits were to already-tracked
    files, not new ones, per the "Add Files to the Repo" rule
