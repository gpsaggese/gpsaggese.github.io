# Build the paper from either of two sources:
#
# - Markdown (`paper.md`): pandoc with Typst (via the official IEEE
#   "charged-ieee" template) as the PDF engine.
# - LaTeX (`paper.tex`): `run_latex.py`, which runs `pdflatex` and `bibtex`
#   inside a Docker container (no local LaTeX installation needed).
#
# Usage:
#   > make                 # build paper_md.pdf from paper.md (default)
#   > make md              # same as above
#   > make latex           # build paper_tex.pdf from paper.tex
#   > make all             # build both PDFs
#   > make figures         # regenerate figures/*.png
#   > make clean           # remove build artifacts
#
# Options for the LaTeX build (see `run_latex.py --help`):
#   > make latex NUM_PASSES=3                    # 3 = with bibliography
#   > make latex RUN_LATEX_ACTIONS=              # also open the PDF (the
#                                                # `run_latex.py` default)
#   > make latex RUN_LATEX_ACTIONS="--action copy_to_gdrive"
#
# The two PDFs have distinct names because `pdflatex` writes `paper.pdf`
# next to `paper.tex` before `run_latex.py` renames it, which would overwrite
# a Typst output called `paper.pdf`.

GIT_ROOT := $(shell git rev-parse --show-toplevel)

# Markdown flow.
PANDOC = pandoc
SRC_MD = paper.md
TEMPLATE = ieee-template.typ
OUT_MD = paper_md.pdf

# LaTeX flow.
RUN_LATEX = $(GIT_ROOT)/helpers_root/dev_scripts_helpers/documentation/run_latex.py
SRC_TEX = paper.tex
OUT_TEX = paper_tex.pdf
# Number of `pdflatex` passes: 1 (quick), 2 (cross-references), 3 (also
# `bibtex`, needs a `\bibliography{references}` command in `$(SRC_TEX)`).
NUM_PASSES = 2
# Run only the compilation: skip opening the PDF, which `run_latex.py` does by
# default. Copying to Google Drive is not a default action. Note that
# `--action` adds to the default actions rather than replacing them.
RUN_LATEX_ACTIONS = --skip_action open_pdf

.PHONY: all md latex figures clean

md: $(OUT_MD)

all: md latex

# Add one prerequisite per figure/table referenced in $(SRC_MD), e.g.
# figures/example_figure.png, so `make` rebuilds the PDF when a figure
# changes.
$(OUT_MD): $(SRC_MD) $(TEMPLATE) references.bib figures/example_figure.png
	$(PANDOC) $(SRC_MD) \
		--from markdown+smart \
		--template $(TEMPLATE) \
		--to typst \
		--pdf-engine typst \
		-o $(OUT_MD)

# `latex` is phony because `run_latex.py` runs inside Docker and tracks
# `\input`ed files itself.
latex:
	$(RUN_LATEX) \
		--input $(SRC_TEX) \
		--output $(OUT_TEX) \
		--num_passes $(NUM_PASSES) \
		$(RUN_LATEX_ACTIONS)

figures: figures/make_figures.py
	python3 figures/make_figures.py

clean:
	rm -f $(OUT_MD) $(OUT_TEX) paper.aux paper.log paper.out paper.toc \
		paper.bbl paper.blg
