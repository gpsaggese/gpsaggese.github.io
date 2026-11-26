#!/bin/bash -xe
# """
# Rebuild the Jupyter Books and serve the website locally.
# """
# Activate the venv with `mkdocs` (create it with `website/create_venv.sh`).
source "$HOME/src/venv/client_venv.website/bin/activate"
cd website
# Rebuild the Jupyter Book tutorials so they are always up to date under
# docs/jupyter_books/ before previewing (see publish_jupyter_books.sh).
./publish_jupyter_books.sh
mkdocs serve --clean --open
