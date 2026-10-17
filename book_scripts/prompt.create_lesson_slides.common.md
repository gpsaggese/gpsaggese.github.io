# Goal
- Describe the steps shared by the prompts that create the slides of one lesson
- Do not use this file alone: it is executed by
  - `book_scripts/prompt.create_lesson_slides_from_chap_toc.md`
  - `book_scripts/prompt.create_lesson_slides_from_topic.md`

# Parameters
- The calling prompt defines
  - `<DIR>`: root of the repo
  - `<BOOK>`: book or course directory (e.g., `book_Modern_AI_for_Finance`,
    `msml610`)
  - `<CHAP_NUM>`: chapter number, two digits
  - `<ID>`: lesson index inside the chapter, two digits (`01` for the first
    lesson)
  - `<TITLE>`: lesson title with words joined by `_`
  - The topics to cover
  - The material already read (existing lessons, papers, books)

- Output file is
  `<DIR>/<BOOK>/lectures_source/Lesson<CHAP_NUM>.<ID>_<TITLE>.smd`
  - If the file already exists, overwrite it

# Workflow

## Create a Skeleton of Slides
- Create a list of slide titles only, about 30-35 slides covering the topics
  - Group the slides under `#` headers
  - Use `##` headers only for a group of at least 5 slides, following "Section
    Structure" in `.claude/skills/slides.rules.md`
- Show the list to the user and wait for approval
  - Skip the approval only if the user asks for it explicitly
- If you have doubts or something is not clear, stop and ask questions before
  proceeding

## Create Slides
- Execute `.claude/skills/slides.write/SKILL.md` with `<NUM_SLIDES>` = 30-35
  using the approved skeleton and the material read by the calling prompt

- Follow all formatting rules, templates, and structural guidelines in
  `.claude/skills/slides.rules.md`
  - Prose and markdown conventions are in `.claude/skills/text.rules.md` and
    `.claude/skills/markdown.rules.md`

- If you reuse slides or material from the files read by the calling prompt,
  add a `// From: ...` comment right after `* <slide title>`
  - Follow "Source Attribution" in `.claude/skills/slides.rules.md`
  - E.g.,
    ```
    // From: msml610/lectures_source/Lesson08.1-Causal_AI_intro.smd:318 '* Where ML Falls Short'
    ```

- When you have a doubt about a claim, number, example, or reference, add a
  `// TODO(gp): Check ...` comment right above the bullet it refers to
  - Doubts include content recalled from memory and not found in the material
    read, illustrations not taken from a source, and unverified metadata of a
    reference (e.g., year, venue)
  - Say what to check and why it is in doubt
  - Start the comment at column 0, even under a sub-bullet
    - The preprocessor strips only `//` lines that start at column 0, so an
      indented comment shows up in the rendered slide
  - E.g.,
    ```
    // TODO(gp): Check that de Prado (2018, ch. 7) suggests h_e = 0.01T: it is
    // recalled from memory and is not in the notes
    - @Definition@: an **embargo** drops $h_e$ observations after each test
      block
    ```
  - List all the `TODO(gp)` comments in the final report to the user

- Do not run the linter on the slides

## Create Notebook
- If there are figures that you need to generate for the slides (e.g., plots of
  simulated or empirical data), create a Jupyter notebook following the specs in
  `.claude/skills/notebook.rules.md` and using the skills in
  `.claude/skills/notebook.*/SKILL.md`
  - Skip this step if no slide needs a figure computed from data or code
  - Diagrams, tables, and portraits do not need a notebook: follow
    `.claude/skills/figure.rules.md`

- Create the notebook in two phases
  - Outline: `.claude/skills/notebook.create_outline/SKILL.md`
  - Implementation: `.claude/skills/notebook.implement_outline/SKILL.md`
  - Generate only the figures required by the slides, do not add extra
    exploration

- Save the notebook files in the directory
  `<DIR>/<BOOK>/tutorials/L<CHAP_NUM>_<TITLE>/`
  - Notebook: `L<CHAP_NUM>_<ID>_<TITLE>.ipynb`
  - Paired Jupytext file: `L<CHAP_NUM>_<ID>_<TITLE>.py`
  - Utility file: `L<CHAP_NUM>_<ID>_<TITLE>_utils.py`
  - Use underscores in all the names and lower case for `<TITLE>`, following
    "Notebook-to-File Pairing" in `.claude/skills/notebook.rules.md`
  - E.g., `msml610/tutorials/L03_knowledge_representation/L03_02_wumpus_world.ipynb`

- Save each figure as a PNG file with `hmatplo.save_fig()` from
  `helpers_root/helpers/hmatplotlib.py`
  - Import it with `import helpers.hmatplotlib as hmatplo`
  - Do not call `fig.savefig()` or `plt.savefig()` directly
  - Plot in a function of the `*_utils.py` file that returns `fig`, then save
    the figure in the notebook cell
  - Use `print_markdown=True` and `path_prefix`, so that the cell prints the
    markdown reference to paste in the slides
  - Name the file `L<CHAP_NUM>.<ID>.<Figure_Name>.png`
  - E.g.,
    ```python
    out_dir_name = "figures"
    markdown_path_prefix = "<BOOK>/lectures_source"

    fig = utils.plot_return_distribution(data)
    hmatplo.save_fig(
        fig,
        f"{out_dir_name}/L01.01.Return_Distribution.png",
        print_markdown=True,
        path_prefix=markdown_path_prefix,
    )
    ```
  - Follow the pattern in
    `msml610/tutorials/L08_causal_inference/L08_04_01_causal_inference.py`
  - Use the color palette in `.claude/skills/figure.rules.md`

- Run the notebook so that all the PNG files exist
  - The PNG files are saved in `figures/` of the notebook working directory
  - Copy them to `<DIR>/<BOOK>/lectures_source/figures/`, as
    `msml610/tutorials/sync_figs.sh` does

- Add the figures to the slides
  - Paste the markdown reference printed by `hmatplo.save_fig()`, a path
    relative to `<DIR>`, e.g.,
    ```
    ![](<BOOK>/lectures_source/figures/L01.01.Return_Distribution.png)
    ```
  - Put the figure next to the text in columns, following "Visuals" in
    `.claude/skills/slides.rules.md`

## Save Results
- Save the slides in the output file described in `# Parameters`
