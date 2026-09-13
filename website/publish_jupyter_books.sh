#!/usr/bin/env bash
# """
# Build the MyST Jupyter Books (`data605`, `msml610`, `tutorials`) and
# publish them as static sub-sites inside the MkDocs website, under
# `website/docs/jupyter_books/<book>/`.
#
# This automates the manual steps documented in
# `msml610/jupyter_book/README.md`: MkDocs copies any non-Markdown files
# under `website/docs/` into the built site as-is (the same mechanism used
# for `website/docs/class_links/*.html`), so publishing a book is just:
# build it with `jupyter-book build --html`, copy its `_build/html/` output
# under `website/docs/jupyter_books/<book>/`, then build/deploy the website
# as usual.
#
# Each book is not added to `nav:` in `website/mkdocs.yml`, so it is
# reachable by direct link but does not appear in the site's top navigation
# (the same treatment given to `class_links`).
#
# Usage:
# - Build and copy all 3 books, then preview locally:
#   > ./website/publish_jupyter_books.sh
#   > cd website && mkdocs serve
#
# - Build and copy only a subset of books:
#   > ./website/publish_jupyter_books.sh --books data605,msml610
#
# - Build, copy, and deploy the website to GitHub Pages in one step:
#   > ./website/publish_jupyter_books.sh --deploy
#
# - Force a rebuild even if no source file changed:
#   > ./website/publish_jupyter_books.sh --force
# """
set -euo pipefail

GIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# All books share one jupyter-book v2 venv (see e.g.
# `data605/jupyter_book/create_venv.sh`).
VENV_DIR="$HOME/src/venv/client_venv.jupyter_book2"
OUT_DIR="$GIT_ROOT/website/docs/jupyter_books"

ALL_BOOKS=("data605" "msml610" "tutorials")
BOOKS=("${ALL_BOOKS[@]}")
DEPLOY=0
FORCE=0

_print_usage() {
  # """
  # Print the script's command-line usage.
  # """
  echo "Usage: $0 [--books data605,msml610,tutorials] [--deploy] [--force]"
}

_book_is_up_to_date() {
  # """
  # Check whether a book's published output is already up to date, i.e. no
  # file it references (per its `myst.yml` `file:` entries) or `myst.yml`
  # itself changed since the last successful publish of this book.

  # :param book: course/book directory name (e.g., "data605")
  # :return: 0 (true) if up to date, 1 (false) otherwise
  # """
  local book="$1"
  local book_dir="$GIT_ROOT/$book/jupyter_book"
  local stamp="$book_dir/_build/.publish_stamp"
  if [[ "$FORCE" -eq 1 || ! -f "$stamp" || ! -d "$OUT_DIR/$book" ]]; then
    return 1
  fi
  # Resolve every `file:` entry in myst.yml relative to $book_dir, plus
  # myst.yml itself, and check if any is newer than the last publish stamp.
  local rel_path abs_path
  while IFS= read -r rel_path; do
    abs_path="$book_dir/$rel_path"
    if [[ -e "$abs_path" && "$abs_path" -nt "$stamp" ]]; then
      return 1
    fi
  done < <(grep -E '^\s*-?\s*file:' "$book_dir/myst.yml" | sed -E 's/^[^:]*:[[:space:]]*//')
  if [[ "$book_dir/myst.yml" -nt "$stamp" ]]; then
    return 1
  fi
  return 0
}

_build_book() {
  # """
  # Build one jupyter-book and copy its static HTML output into the website.
  # Skips the (re)build entirely if the output is already up to date, unless
  # --force was passed.

  # :param book: course/book directory name (e.g., "data605")
  # """
  local book="$1"
  local book_dir="$GIT_ROOT/$book/jupyter_book"
  if [[ ! -d "$book_dir" ]]; then
    echo "ERROR: no jupyter_book dir at $book_dir" >&2
    exit 1
  fi
  if _book_is_up_to_date "$book"; then
    echo "==> $book jupyter book is up to date, skipping (use --force to rebuild)"
    return
  fi
  echo "==> Building $book jupyter book"
  (
    cd "$book_dir"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    # mystmd writes every asset/page link as a root-absolute path
    # (e.g. `/build/...`, `/favicon.ico`). Since the book is served from
    # `/jupyter_books/<book>/` and not the domain root, BASE_URL must be set
    # at build time so those links resolve correctly. See:
    # https://mystmd.org/guide/deployment-github-pages
    BASE_URL="/jupyter_books/$book" jupyter-book build --html
  )
  echo "==> Copying $book build output to $OUT_DIR/$book"
  mkdir -p "$OUT_DIR"
  rm -rf "${OUT_DIR:?}/$book"
  cp -r "$book_dir/_build/html" "$OUT_DIR/$book"
  # mystmd embeds a raw copy of every source notebook/markdown file under
  # `build/` as a "view source" download link. These are the single biggest
  # contributor to the published size (~120MB across all books) and the
  # `.md` ones already 404 in practice (MkDocs renders any `.md` it finds
  # into its own page instead of copying it verbatim), so drop both from
  # what gets published. This does disable the "download .ipynb" link on
  # the live site.
  echo "==> Dropping raw .ipynb/.md source copies from $OUT_DIR/$book"
  find "$OUT_DIR/$book" -type f \( -iname '*.ipynb' -o -iname '*.md' \) -delete
  # Mark this book as freshly published so the next run can skip it if
  # nothing changed (see _book_is_up_to_date).
  touch "$book_dir/_build/.publish_stamp"
}

# Parse command-line args.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --books)
      IFS=',' read -r -a BOOKS <<< "$2"
      shift 2
      ;;
    --deploy)
      DEPLOY=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      _print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      _print_usage
      exit 1
      ;;
  esac
done

# Ensure the shared jupyter-book venv exists before building anything.
if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR" >&2
  echo "Run e.g. ./data605/jupyter_book/create_venv.sh first." >&2
  exit 1
fi

# Build and copy each requested book.
for book in "${BOOKS[@]}"; do
  _build_book "$book"
done

# Optionally build/deploy the whole website (books included) to GitHub Pages.
if [[ "$DEPLOY" -eq 1 ]]; then
  echo "==> Deploying website to GitHub Pages"
  # --no-history squashes the gh-pages branch to a single commit on every
  # deploy instead of appending one: without it, republishing these large
  # generated books would grow the gh-pages branch's stored history forever.
  (cd "$GIT_ROOT/website" && mkdocs gh-deploy --no-history)
  open "https://gpsaggese.github.io/" 2>/dev/null || true
else
  cat <<EOF
==> Done. Books copied under $OUT_DIR

Preview locally:
  > cd website && mkdocs serve
  # or
  > ./website/preview_website.sh

Deploy to GitHub Pages:
  > ./website/publish_jupyter_books.sh --deploy
  # or
  > cd website && mkdocs gh-deploy
EOF
fi
