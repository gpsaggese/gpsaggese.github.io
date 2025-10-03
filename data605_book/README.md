https://jupyterbook.org/en/stable/start/create.html

Create env
> python3 -m venv ~/src/venv/client_venv.jupyter_book
> source ~/src/venv/client_venv.jupyter_book/bin/activate
> pip install -r data605_book/requirements.txt

> jupyter-book create data605_book/

> ls -1 data605_book/*

| data605_book/\_config.yml | Configuration |
| :---- | :---- |
| data605_book/\_toc.yml | Structure / TOC |
| data605_book/intro.md | md |
| data605_book/markdown-notebooks.md |  |
| data605_book/markdown.md |  |
| data605_book/notebooks.ipynb |  |
| data605_book/references.bib |  |
| data605_book/requirements.txt |  |

> jupyter-book build data605_book/

> open file:///Users/saggese/src/umd_classes1/data605_book/_build/html/index.html

