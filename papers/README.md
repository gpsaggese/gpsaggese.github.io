- Every paper dir has a `Makefile` that is a symlink to `template/Makefile`, so
  that all the dirs stay in sync
  - The `Makefile` builds `paper.md` (pandoc + Typst) and / or `paper.tex`
    (`run_latex.py`, which also renders the diagrams with `render_images.py`)
  - Customize a build from the command line, not by editing the `Makefile`
  - See the header of `template/Makefile` for all the options

- Build a paper from the top of the repo
  ```
  > cd $GIT_ROOT
  > make -C papers/gp_saggese_cv
  > make -C papers/gp_saggese_quant_research_cv
  ```
  - The output is `paper.pdf` in the paper dir

- Lint a paper
  ```
  > lint_text.py -i papers/gp_saggese_cv/paper.tex --use_dockerized_prettier
  ```
