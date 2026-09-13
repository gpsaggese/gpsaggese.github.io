# How to Work on Interactive Notebooks

- This document explains how a tutorial notebook (`msml610/tutorials/L<NN>_<topic>/*.ipynb`)
  turns into something a reader can either jump to from the slides or run live
  A. A reader jump from a slide straight to the matching notebook cell (view, maybe
     run)
  B. A reader run the notebook themselves without setting up a local environment
  C. A reader runs the notebook locally using Docker

- The two goals need different hosting, and can be combined: a Colab link is
    both clickable from a slide (with `#scrollTo=<cell_id>`) and runnable

## Command Summary

| Command/Tool                                               | Purpose                                                        |
| :----------------------------------------------------------| :--------------------------------------------------------------|
| `jupyter nbconvert --to html`                               | Convert a notebook to a static HTML file                       |
| `jupyter nbconvert --to html --execute`                     | Convert to HTML and re-run all cells, baking in fresh outputs  |
| `helpers_root/dev_scripts_helpers/notebooks/publish_notebook.py` | Convert/publish/open a notebook as HTML (local, S3, or webserver) |
| `helpers_root/dev_scripts_helpers/notebooks/extract_notebook_images.py` | Extract screenshots from marked notebook cells           |
| `helpers_root/dev_scripts_helpers/notebooks/add_toc_to_notebook.py` | Add a clickable table of contents inside the notebook itself |
| `helpers_root/dev_scripts_helpers/notebooks/run_nbconvert.sh` | Execute a notebook top to bottom and convert to HTML inside Docker, with `html_anchorfix` |
| `claude> /slides.add_tutorial_links $SMD_FILE`               | Link each slide section to its matching notebook cell           |
| `class_scripts/colab_setup.py`                               | Shared setup so a tutorial notebook's `helpers`/`_utils.py` imports work on Colab/Binder |

## Flow 1: Static HTML with Per-Cell Anchors (for slide -> cell links)

- Use this when the only goal is Goal A (jump from a slide to the right spot in
  the notebook), read-only is fine

### Prepare the Notebook

- If the notebook uses `ipywidgets`, turn on widget-state saving so the static
  export still shows the widgets' last rendered state:
  - Jupyter: `Settings -> Save Widget State Automatically`
  - Re-run all cells afterward
- Without this step, cells with interactive widgets export as blank in the HTML

### Convert to HTML with Working Cell Anchors

```bash
> jupyter nbconvert --to html \
  --template html_anchorfix \
  --TemplateExporter.extra_template_basedirs=<repo_root>/helpers_root/dev_scripts_helpers/notebooks/nbconvert_templates \
  <notebook>.ipynb
```

- The `html_anchorfix` template lives at
  `helpers_root/dev_scripts_helpers/notebooks/nbconvert_templates/html_anchorfix/`
- Plain `jupyter nbconvert --to html` (no template) works too, but the per-cell
  heading anchors are not reliable, use `html_anchorfix` whenever you need to
  link to a specific cell
- The anchor for a markdown cell heading is
  `#Cell-N:-Cell-Title-With-Dashes-For-Spaces`, e.g.
  `#Cell-1:-Introduction---Casino-Slot-Machines`

### Host the HTML

- `https://raw.githack.com/<user>/<repo>/<branch>/<path>.html` works as a
  CDN-fronted raw view of the file, use this for a scratch/WIP branch
- `https://htmlpreview.github.io/?https://github.com/...` does **not** work for
  these exported files, do not use it
- `gpsaggese.github.io` (this repo) is already a GitHub Pages site: once the HTML
  file is merged into the branch Pages serves (usually `main`), it is live
  directly at `https://gpsaggese.github.io/<path>.html`, no `raw.githack` needed

### Link the Slides to the Notebook

- Run `claude> /slides.add_tutorial_links $SMD_FILE` with the hosted HTML URL to
  automatically match each slide section to a notebook cell and insert the anchor

## Flow 2: Interactive / Runnable Notebook (Goal B)

- Use this when a reader should be able to edit and run the code, not just read it

### Make the Notebook's Own Imports Work on Colab/Binder

- Colab and Binder only ever see the single notebook file, not the whole repo,
  so `import helpers.hnotebook` and the notebook's own paired `_utils.py`
  import fail there (`ModuleNotFoundError: No module named 'helpers'`) unless
  the notebook fetches them itself first
- `helpers_root` is also a **git submodule**
  (`git@github.com:causify-ai/helpers.git`): a plain `git clone`/`git pull` of
  the repo leaves it on disk but empty, and its pinned URL is SSH, which needs
  a key neither Colab nor Binder has
- `class_scripts/colab_setup.py` is the shared, stdlib-only fix for all of
  this (repo clone on Colab, submodule fetch over HTTPS, `sys.path`/cwd
  wiring, `requirements.txt` install, autoreload gating), reusable by any
  `msml610`/`data605` tutorial notebook: only the `setup()` argument
  changes per notebook, everything else is copy-paste
- Add this as the **first code cell** of the notebook, before any
  `helpers`/`*_utils` import and before `%load_ext autoreload`:

  ```python
  import os
  import sys

  ON_COLAB = "google.colab" in sys.modules
  ON_BINDER = "BINDER_LAUNCH_HOST" in os.environ

  if ON_COLAB or ON_BINDER:
      import subprocess

      # `colab_setup` lives inside the repo, so on Colab it isn't
      # importable until the repo is cloned; Binder and local runs
      # already have it, just not always on `sys.path`.
      if ON_COLAB and not os.path.exists("gpsaggese.github.io"):
          subprocess.run(
              [
                  "git", "clone", "--depth", "1", "--branch", "gp",
                  "https://github.com/gpsaggese/gpsaggese.github.io.git",
              ],
              check=True,
          )
      repo_root = (
          os.path.abspath("gpsaggese.github.io")
          if ON_COLAB
          else subprocess.run(
              ["git", "rev-parse", "--show-toplevel"],
              capture_output=True, text=True, check=True,
          ).stdout.strip()
      )
      # Docker gets this for free via PYTHONPATH; Colab/Binder need it.
      sys.path.insert(0, repo_root)

  import class_scripts.colab_setup as colab_setup

  colab_setup.setup("msml610/tutorials/<LNN_topic>")
  colab_setup.maybe_enable_autoreload()
  ```

- `colab_setup.maybe_enable_autoreload()` replaces a bare `%load_ext
  autoreload` / `%autoreload 2`: it skips them on Colab/Binder (nothing to
  watch until `setup()` has run) and outside a notebook kernel (the paired
  `.py` run as a plain script, where `get_ipython()` is `None` and a bare
  `%`-magic would be a syntax error)
- On Colab, a rerun in the same session `git pull`s instead of re-cloning, so
  a later push is picked up, but already-imported modules stay cached:
  **Runtime -> Restart session**, then **Run all**, to actually see the update
- Worked examples:
// TODO(ai_gp): Update these links
  - `msml610/tutorials/L03_knowledge_representation/L03_01_entailment_implication_inference.py`
    for the Binder-only variant (`if ON_BINDER:`), used when a notebook does
    not need to run on Colab
  - `msml610/tutorials/L03_knowledge_representation/examples/L03_01_entailment_implication_inference_colab.py`
    for the Colab-only variant (`if ON_COLAB:`), a standalone copy kept in an
    `examples/` dir specifically for a Colab link

### Option 1: Google Colab (Recommended)

- Push the notebook to GitHub, e.g., the `gp` branch of this repo
- Add a link to the notebook's first cell:

  ```markdown
  [Open in Google Colab](https://colab.research.google.com/github/gpsaggese/gpsaggese.github.io/blob/gp/<path>.ipynb)
  ```

- URL format: `https://colab.research.google.com/github/<user>/<repo>/blob/<branch>/<path>.ipynb`
- Clicking the badge opens a live, runnable copy in Colab: free CPU/GPU, needs a
  Google account
- To deep-link a **specific cell**, append `#scrollTo=<cell_id>`
  - `<cell_id>` is the cell's own `id` field in the `.ipynb` JSON (nbformat >= 4.5
    assigns one to every cell automatically), read it directly from the file,
    no need to open the notebook in Colab first:

    ```bash
    > python3 -c "
    import json
    nb = json.load(open('<notebook>.ipynb'))
    for c in nb['cells']:
        print(c['cell_type'], c['id'], ''.join(c.get('source', []))[:60])
    "
    ```

  - E.g.,
    `https://colab.research.google.com/github/gpsaggese/gpsaggese.github.io/blob/gp/msml610/tutorials/L03_knowledge_representation/examples/L03_01_entailment_implication_inference_colab.ipynb#scrollTo=c1a95a7b`

### Option 2: Binder

- No link edit needed on the notebook itself
- Go to `https://mybinder.org`, enter the repo URL, branch, and notebook path,
  get a shareable link, or embed a link:

  ```markdown
  [Open in Binder](https://mybinder.org/v2/gh/gpsaggese/gpsaggese.github.io/gp?filepath=<path>.ipynb)
  ```

- Builds the full environment from the tutorial's `requirements.txt`, so it
  matches the repo's dependencies exactly
- Slower first launch (environment build), cached afterward

### Option 3: nbviewer

- `https://nbviewer.org/github/<user>/<repo>/blob/<branch>/<path>.ipynb`
- Static rendering only, nicer than GitHub's own notebook viewer, not
  interactive
- Skip it if readers need to run code, use Colab or Binder instead

### Baked-Output Static Export

```bash
> jupyter nbconvert --to html --execute <notebook>.ipynb
```

- Re-runs every cell and bakes the outputs into the HTML, then host it (GitHub
  Pages, S3, etc.) the same way as Flow 1
- Good for a static report a reader does not need to edit or run
- Not a substitute for Colab/Binder if readers should be able to edit/run the
  code

## Which Option to Use

| Need                                                    | Use                                   |
| :-------------------------------------------------------| :--------------------------------------|
| Jump from a slide straight to a runnable cell            | Colab link with `#scrollTo=<cell_id>` |
| Match the repo's exact dependencies, can wait on a build | Binder                                |
| Quick, read-only preview, no interactivity needed        | `raw.githack` HTML or nbviewer        |
| A static report with outputs baked in                    | `nbconvert --to html --execute`       |
| Auto-link an entire lecture's slides to a tutorial        | `claude> /slides.add_tutorial_links`  |
