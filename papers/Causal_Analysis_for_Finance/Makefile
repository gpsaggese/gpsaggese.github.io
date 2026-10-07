# Build the paper from LaTeX (`paper.tex`) with `run_latex.py`, which runs
# `pdflatex` and `bibtex` inside a Docker container (no local LaTeX
# installation needed).
#
# Usage:
#   > make                 # build paper.pdf from paper.tex
#   > make figures         # regenerate figures/*.png
#   > make clean           # remove build artifacts
#
# Options (see `run_latex.py --help`):
#   > make NUM_PASSES=2                          # 2 = without bibliography
#   > make RUN_LATEX_ACTIONS=                    # also open the PDF (the
#                                                # `run_latex.py` default)
#   > make RUN_LATEX_ACTIONS="--action copy_to_gdrive"

GIT_ROOT := $(shell git rev-parse --show-toplevel)

RUN_LATEX = $(GIT_ROOT)/helpers_root/dev_scripts_helpers/documentation/run_latex.py
SRC = paper.tex
OUT = paper.pdf
# Number of `pdflatex` passes: 1 (quick), 2 (cross-references), 3 (also
# `bibtex`, needs a `\bibliography{references}` command in `$(SRC)`).
NUM_PASSES = 3
# Run only the compilation: skip opening the PDF, which `run_latex.py` does by
# default. Copying to Google Drive is not a default action. Note that
# `--action` adds to the default actions rather than replacing them.
RUN_LATEX_ACTIONS = --skip_action open_pdf

.PHONY: all figures clean

all: $(OUT)

# `run_latex.py` runs inside Docker and tracks `\input`ed files itself, so the
# rule depends on the sources it reads.
$(OUT): $(SRC) style.tex references.bib
	$(RUN_LATEX) \
		--input $(SRC) \
		--output $(OUT) \
		--num_passes $(NUM_PASSES) \
		$(RUN_LATEX_ACTIONS)

figures: figures/make_figures.py
	python3 figures/make_figures.py

clean:
	rm -f $(OUT) paper.aux paper.log paper.out paper.toc paper.bbl paper.blg
