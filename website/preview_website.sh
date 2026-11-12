#!/bin/bash -xe
cd website
# Rebuild the Jupyter Book tutorials so they are always up to date under
# docs/jupyter_books/ before previewing (see publish_jupyter_books.sh).
./publish_jupyter_books.sh
mkdocs serve --clean --open
