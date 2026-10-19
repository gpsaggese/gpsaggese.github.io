---
title: "Render Diagrams in Markdown and LaTeX with render_images.py"
draft: true
authors:
    - gpsaggese
date: 2026-10-19
description:
    "Use render_images.py to turn version-controlled diagram blocks into images
    while keeping their editable source in the document."
categories:
    - Developer Tools
    - Python
---

TL;DR: `render_images.py` finds supported diagram blocks in Markdown, LaTeX,
text, and Typst files, renders them as images, and inserts image references
while preserving the source in comments. Run it when you need a rendered
document and want the diagram code to stay version-controlled alongside it.

<!-- more -->

## The problem it solves

A diagram often has two copies: editable source code and a rendered image. If
those are stored separately, they can drift apart. `render_images.py` keeps the
source block in the document and regenerates the image from it, so a later edit
has a clear source of truth.

The tool is part of the
[Causify helpers repository](https://github.com/causify-ai/helpers/blob/17e42d8115a8c033c88edc290082da5843b6cd05/dev_scripts_helpers/documentation/render_images.py).
It recognizes fenced blocks labeled `plantuml`, `mermaid`, `tikz`, `graphviz`,
`latex`, `raw_latex`, `svg`, or `image`.

## Render a diagram

For example, put a Mermaid block in a Markdown file:

````markdown
```mermaid
flowchart LR
    A[Write source] --> B[Render image]
    B --> C[Keep both in version control]
```
````

From the repository environment, render it in place:

```bash
> ./helpers_root/dev_scripts_helpers/documentation/render_images.py \
    -i README.md --action render --run_dockerized
```

The tool creates a `README.md.figs/` directory next to the Markdown file, writes
an image there, replaces the active diagram block with an image reference, and
keeps the original block in an HTML comment. The generated image name includes
the source file name and block number, which keeps paths predictable as you
rerender the same document.

`--run_dockerized` runs the renderer in the project's Docker environment. If you
omit it, the required rendering tools must be installed locally. The output is
PNG by default; use `--output_format svg` for supported diagram types when you
want a scalable image. TikZ and LaTeX blocks currently require PNG output.

## Preview, rerender, or restore the source

To open a preview of the rendered figures in a browser without rendering again:

```bash
> ./helpers_root/dev_scripts_helpers/documentation/render_images.py \
    -i README.md --clear_actions --action open
```

If you change a diagram, run the `render` command again. The renderer caches
image work, so unchanged source can reuse a prior result. To remove generated
figure references and uncomment the original diagram code:

```bash
> ./helpers_root/dev_scripts_helpers/documentation/render_images.py \
    -i README.md --remove_figs
```

Use `--output other.md` to write a separate Markdown output instead of editing
the input file. For multiple files, the CLI supports a comma-separated `--files`
list, a `--from_files` list file, or repeated `--input` arguments; multi-file
rendering happens in place.

## What it supports

| Source block                 | Common use                                                  |
| :--------------------------- | :---------------------------------------------------------- |
| `mermaid`                    | Flowcharts, sequence diagrams, and timelines                |
| `plantuml`                   | UML and interaction diagrams                                |
| `graphviz`                   | Directed and undirected graphs                              |
| `tikz`, `latex`, `raw_latex` | Technical figures and mathematical material                 |
| `svg`                        | Existing SVG markup                                         |
| `image`                      | A prompt passed to the repository's image-generation helper |

The command accepts `.md`, `.tex`, `.txt`, and `.typ` documents. The renderer
inserts a reference using the syntax for the source document. For Markdown and
LaTeX, that means Markdown image markup or `\\includegraphics`; Typst uses a
`figure` with an `image(...)` call. The
[tool README](https://github.com/causify-ai/helpers/blob/17e42d8115a8c033c88edc290082da5843b6cd05/dev_scripts_helpers/documentation/render_images.README.md)
has current options and format-specific details.

## When it is a good fit

Use the tool when the diagram source should live next to the prose, when the
same document needs rendered figures for readers, or when a repeatable
documentation build should regenerate figures. It can also help turn embedded
LaTeX or TikZ source into figures without maintaining separate exports.

It is less useful for interactive diagrams or for images whose editable source
does not exist in the document. The output is a static image. Image blocks also
invoke the repository's image-generation helper, which requires its own
configuration; they are not ordinary local diagram rendering.

## Complete example and source

The source code, tests, and usage notes are maintained in the helpers
repository:

- [Implementation](https://github.com/causify-ai/helpers/blob/17e42d8115a8c033c88edc290082da5843b6cd05/dev_scripts_helpers/documentation/render_images.py)
- [Tests](https://github.com/causify-ai/helpers/blob/17e42d8115a8c033c88edc290082da5843b6cd05/dev_scripts_helpers/documentation/test/test_render_images.py)
- [Tool README](https://github.com/causify-ai/helpers/blob/17e42d8115a8c033c88edc290082da5843b6cd05/dev_scripts_helpers/documentation/render_images.README.md)

One practical habit: keep the editable code block in the document, rerender
after changing it, and review both the diff and the output image before sharing
the document.
