# Build a paper from Markdown (`paper.md`) and / or LaTeX (`paper.tex`).
#
# This is the single Makefile of all the paper dirs: every `papers/*/Makefile`
# is a symlink to `papers/template/Makefile`, so they stay in sync. It detects
# what to build from the files in the current dir, so it must not contain
# paper-specific settings: pass them on the command line instead.
#
# - Markdown (`paper.md`): pandoc with Typst (via the official IEEE
#   "charged-ieee" template, `ieee-template.typ`) as the PDF engine.
# - LaTeX (`paper.tex`): `run_latex.py`, which renders the diagrams with
#   `render_images.py`, then runs `pdflatex` and `bibtex` inside a Docker
#   container (no local LaTeX installation needed).
#
# Usage:
#   > make                 # build the PDF from `paper.md` and / or `paper.tex`
#   > make md              # build the PDF from `paper.md`
#   > make latex           # build the PDF from `paper.tex`
#   > make figures         # regenerate figures/*.png
#   > make clean           # remove build artifacts
#   > make -C papers/Noesis   # build a paper from the top of the repo
#
# The PDF is `paper.pdf`. If both `paper.md` and `paper.tex` exist (e.g., in
# `papers/template`), the PDFs are `paper_md.pdf` and `paper_tex.pdf`, since
# `pdflatex` writes `paper.pdf` next to `paper.tex` before `run_latex.py`
# renames it, which would overwrite a Typst output called `paper.pdf`.
#
# Options for the LaTeX build (see `run_latex.py --help`):
#   > make latex NUM_PASSES=3                    # 3 = with bibliography
#   > make latex RUN_LATEX_ACTIONS=              # also open the PDF (the
#                                                # `run_latex.py` default)
#   > make latex RUN_LATEX_ACTIONS="--action copy_to_gdrive"
#
# `make` rebuilds a PDF only when a source changed, e.g., `paper.md`,
# `references.bib`, `*.tex`, `figures/*.png`. Use `make -B` to force it.

GIT_ROOT := $(shell git rev-parse --show-toplevel)

SRC_MD := $(wildcard paper.md)
SRC_TEX := $(wildcard paper.tex)

ifneq ($(and $(SRC_MD),$(SRC_TEX)),)
OUT_MD = paper_md.pdf
OUT_TEX = paper_tex.pdf
else
OUT_MD = paper.pdf
OUT_TEX = paper.pdf
endif

# Files that a build depends on.
BIB = $(wildcard references.bib)
FIGURES = $(wildcard figures/*.png)

# Markdown flow.
PANDOC = pandoc
TEMPLATE = $(firstword $(wildcard ieee-template.typ templates/ieee-template.typ))

# LaTeX flow.
RUN_LATEX = $(GIT_ROOT)/helpers_root/dev_scripts_helpers/documentation/run_latex.py
# The papers share `papers/template/style.tex` through `\input`.
TEX_DEPS = $(wildcard *.tex ../template/style.tex)
# Number of `pdflatex` passes: 1 (quick), 2 (cross-references), 3 (also
# `bibtex`, needs a `\bibliography{...}` command in a `.tex` file, which is
# detected here).
NUM_PASSES = $(if $(shell grep -ls '^[^%]*\\bibliography{' *.tex),3,2)
# Run only the compilation: skip opening the PDF, which `run_latex.py` does by
# default. Copying to Google Drive is not a default action. Note that
# `--action` adds to the default actions rather than replacing them.
RUN_LATEX_ACTIONS = --skip_action open_pdf

.PHONY: all md latex figures clean

all: $(if $(SRC_MD),md) $(if $(SRC_TEX),latex)

ifneq ($(SRC_MD),)
md: $(OUT_MD)

# `$(FIGURES)` makes `make` rebuild the PDF when a figure changes.
$(OUT_MD): $(SRC_MD) $(TEMPLATE) $(BIB) $(FIGURES)
	$(PANDOC) $(SRC_MD) \
		--from markdown+smart \
		--template $(TEMPLATE) \
		--to typst \
		--pdf-engine typst \
		-o $(OUT_MD)
endif

ifneq ($(SRC_TEX),)
latex: $(OUT_TEX)

# `run_latex.py` runs inside Docker and tracks `\input`ed files itself, so the
# rule depends on all the `.tex` files it can read.
$(OUT_TEX): $(TEX_DEPS) $(BIB) $(FIGURES)
	$(RUN_LATEX) \
		--input $(SRC_TEX) \
		--output $(OUT_TEX) \
		--num_passes $(NUM_PASSES) \
		$(RUN_LATEX_ACTIONS)
endif

figures: figures/make_figures.py
	python3 figures/make_figures.py

clean:
	rm -f $(OUT_MD) $(OUT_TEX) paper.aux paper.log paper.out paper.toc \
		paper.bbl paper.blg
