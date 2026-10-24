---
title: "How to Convert PDF to Markdown in Python"
draft: true
authors:
    - gpsaggese
date: 2026-10-23
description:
categories:
    - Developer Tools
    - Python
---

TL;DR: PDFs store layout, not meaning, so converting them to Markdown means
rebuilding the structure; this guide covers four libraries, with installation
steps, usage, and examples for each.

<!-- more -->

## Why Convert PDFs to Markdown?

- PDFs are layout-focused, not semantic: preserve visual structure, lose meaning
- Markdown is portable: version control friendly, tool independent,
  human-readable
- Markdown preserves structure: headings, lists, tables and code marked as such
- LLMs work better with Markdown: cleaner input, better parsing, easier to
  fine-tune
- Single-source publishing: convert once, render to HTML, PDF, EPUB
- Automation: batch convert documents, integrate with pipelines like RAG

## Quick Comparison

| Library      | Speed                                    | Accuracy                                   | Learning Curve                           | Best For                   |
| :----------- | :--------------------------------------- | :----------------------------------------- | :--------------------------------------- | :------------------------- |
| `pymupdf`    | <span style="color:green">Fast</span>    | <span style="color:green">Good</span>      | <span style="color:green">Easy</span>    | Simple docs, speed         |
| `marker-pdf` | <span style="color:yellow">Medium</span> | <span style="color:green">Excellent</span> | <span style="color:yellow">Medium</span> | Academic, structured       |
| `markitdown` | <span style="color:yellow">Medium</span> | <span style="color:green">Good</span>      | <span style="color:green">Easy</span>    | Multi-format (DOCX,PPTX)   |
| `pdfplumber` | <span style="color:red">Slow</span>      | <span style="color:red">Manual</span>      | <span style="color:red">Hard</span>      | Custom logic, full control |

## Installation

**Quick start without installing:**

- You can use `uvx` to run any of these tools without installation
- Ideal for one-off conversions or testing different libraries
- Trade cold-start time (package download) for zero disk footprint
- See each library's "Using uvx" subsection for examples

### pymupdf

On October 19, 2026, `PyMuPDF 2.0` was released and the Markdown converter,
`pymupdf4llm` was integrated into it.

- Install package:

    ```bash
    > pip install pymupdf
    ```

- Verify installation (via CLI):
    ```bash
    # Print the version to verify command is available.
    > python -c "import pymupdf; print(pymupdf.__version__)"
    ```

### Marker-pdf

- Install package:
    ```bash
    > pip install marker-pdf
    ```
- Verify installation (via CLI):
    ```bash
    # Run help to verify command is available.
    > marker_single --help
    ```

### markitdown

- Install package:
    ```bash
    > pip install 'markitdown[pdf]'
    ```
- Verify installation (via CLI):
    ```bash
    # Run help to verify command is available.
    > markitdown --help
    ```

### pdfplumber

- Install package:
    ```bash
    > pip install pdfplumber
    ```
- Verify installation (via Python):
    ```bash
    > python -c "import pdfplumber; print(pdfplumber.__version__)"
    0.11.10
    ```
- Note: `pdfplumber` is primarily a library without a CLI
    - Use Python import verification above
    - Always accessed via Python scripts, not command-line directly

## Basic Usage

### pymupdf: Fast and Simple

- Best for: Speed and simplicity, LLM-friendly output
- Designed for large language models, uses a built-in layout model to find
  headings, tables and reading order

**Python usage:**

- Convert PDF to Markdown:
    ```python
    import pymupdf

    with pymupdf.open("input.pdf") as doc:
      markdown = doc.to_markdown()

    with open("output.md", "w") as f:
        f.write(markdown)
    ```

**Legacy CLI:**

- The latest release with its own CLI is `pymupdf4llm` 1.28.2

**Using uvx (no installation needed):**

- Run directly with `uvx`:
    ```bash
    # Convert PDF without installing.
    > uvx pymupdf4llm@1.28.2 input.pdf --out output-directory-name
    ```
    - Convenient for one-off conversions, trades cold-start time for zero setup

### Marker-pdf: High-Fidelity Layout Detection

- Best for: Academic papers, structured documents, tables
- Uses a vision machine learning model for layout, headings, tables, figures
- It is slower but more accurate on complex pages than text-based tools

**CLI usage:**

- Convert with layout detection:
    ```bash
    # Convert input.pdf with layout analysis.
    > marker_single input.pdf --output_format markdown --output_dir directory_name
    # Convert a whole folder of pdf files into md
    > marker folder_name --output_dir output_dir_name
    ```

**Using uvx (no installation needed):**

- Run directly with `uvx`:
    ```bash
    # Convert with ML detection without installing.
    > uvx --from marker-pdf marker_single input.pdf --output_format markdown --output_dir directory_name
    ```

**Python usage:**

- Convert with ML-based detection:
    ```python
    from pathlib import Path
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import save_output

    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered =   converter("input.pdf")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    save_output(
        rendered,
        str(output_dir),
        "input"
    )
    ```

### markitdown: Multi-Format Support

- Best for: Mixed document types (PDF, DOCX, PPTX)
- Prioritizes consistency across formats

**CLI usage:**

- Convert multiple formats:

    ```bash
    # To use markitdown to convert into multiple formats like DOCX, PPTX execute,
    > pip install 'markitdown[all]'
    # Convert PDF to Markdown.
    > markitdown input.pdf -o output.md
    # Convert DOCX to Markdown.
    > markitdown input.docx -o output.md
    # Convert PPTX to Markdown.
    > markitdown input.pptx -o output.md
    ```

**Using uvx (no installation needed):**

- Run directly with `uvx`:
    ```bash
    # Convert without installing.
    > uvx --from 'markitdown[pdf]' markitdown input.pdf -o output.md
    ```
- Works for all supported formats via single `uvx` call, (change [pdf] to
  intended format as above)

**Python usage:**

- Convert with format detection:
    ```python
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert("input.pdf")
    with open("output.md", "w") as f:
        f.write(result.markdown)
    ```

### pdfplumber: Full Control

- Best for: Custom extraction where you need full control over what is extracted
- Low-level tool: you build the Markdown conversion

**Python usage:**

- Manual extraction with layout control:

    ```python
    import pdfplumber

    markdown_lines = []

    with pdfplumber.open("input.pdf") as pdf:
        for page in pdf.pages:
            # Extract text with layout preservation.
            text = page.extract_text()
            if text:
                markdown_lines.append(text)

    with open("output.md", "w") as f:
        f.write("\n\n".join(markdown_lines))
    ```

- Extract tables:
    ```python
    import pdfplumber

    with pdfplumber.open("input.pdf") as pdf:
        for page in pdf.pages:
            # Find and extract tables.
            tables = page.extract_tables()
            for table in tables:
                print(table)
    ```

## Advanced Features

### Custom Extraction with pdfplumber

- Extract specific page ranges:

    ```python
    import pdfplumber

    with pdfplumber.open("input.pdf") as pdf:
        # Extract pages 1-5.
        for page in pdf.pages[0:5]:
            text = page.extract_text()
            print(text)
    ```

- Crop and extract regions:
    ```python
    import pdfplumber

    with pdfplumber.open("input.pdf") as pdf:
        page = pdf.pages[0]
        # Crop to top-left quadrant.
        cropped = page.crop((0, 0, page.width/2, page.height/2))
        text = cropped.extract_text()
        print(text)
    ```

### Batch Processing

- Convert multiple PDFs:
    ```python
    from pymupdf import convert_batch

    results = convert_batch(
        ["sample.pdf", "sample2.pdf"],
        workers=2,
        out_dir="output",
        backend="md",
    )

    for item in results:
        if item.success:
            print(item.input, "converted successfully")
        else:
            print(item.input, "failed:", item.error)
    ```

### Integration with LLM Processing

- Convert and send to LLM:
    ```python
    import pymupdf

    with pymupdf.open("document.pdf") as doc:
      md = doc.to_markdown()

    # Process with language model.
    from anthropic import Anthropic

    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": f"Summarize:\n{md}"}
        ],
    )
    print(response.content[0].text)
    ```

## Specialized Tool: [convert_pdf_to_md.py](https://github.com/causify-ai/helpers/blob/master/dev_scripts_helpers/documentation/convert_pdf_to_md.py) Script

The `convert_pdf_to_md.py` script in
`helpers_root/dev_scripts_helpers/documentation/` converts PDFs to Markdown,
extracting images and detecting headings along the way.

**Features:**

- Extracts text from PDFs and converts to Markdown with proper heading levels
- Automatically detects and extracts images, saves to `images/` subdirectory
- Analyzes font sizes to distinguish `h1`, `h2`, `h3` headings from body text
- Detects vector graphics and renders them as images
- Preserves image positioning based on PDF layout
- Applies prettier formatting for clean Markdown output
- Supports verbose logging for debugging conversions
- Automatically manages image duplicates via xref tracking
- Uses `uv` for dependency management (no installation needed)

**Installation:**

- Script is self-contained with `uv`:
    ```bash
    # Runs with uv, installs dependencies automatically.
    > ./helpers_root/dev_scripts_helpers/documentation/convert_pdf_to_md.py \
      --input document.pdf \
      --output output_dir
    ```
- Or call via `uv run`:
    ```bash
    # Use uv to run the script.
    > uv run ./helpers_root/dev_scripts_helpers/documentation/convert_pdf_to_md.py \
      --input document.pdf \
      --output output_dir
    ```

**Usage:**

- Basic conversion:
    ```bash
    # Convert PDF to Markdown with images.
    > ./helpers_root/dev_scripts_helpers/documentation/convert_pdf_to_md.py\
      --input input.pdf \
      --output output_dir
    ```
- With verbose logging to debug conversion:
    ```bash
    # See detailed processing information.
    > ./helpers_root/dev_scripts_helpers/documentation/convert_pdf_to_md.py \
      --input input.pdf \
      --output output_dir \
      -v DEBUG
    ```

**Output Structure:**

- Creates output directory with:
    - `output_dir/input.md`: Converted Markdown file
    - `output_dir/images/`: Directory containing all extracted images
        - Page images: `page_1_img_1.png`, `page_1_img_2.jpg`, etc.
        - Rendered pages: `page_1_rendered_1.png` (for vector graphics)

**Example Output:**

- Input PDF: `research.pdf`
- Output structure:
    ```text
    output_dir/
    ├── research.md
    └── images/
        ├── page_1_img_1.png
        ├── page_2_img_1.jpg
        └── page_3_rendered_1.png
    ```

**When to Use convert_pdf_to_md.py:**

- PDFs with mixed text, tables, and images
- Academic papers or technical documents with vector graphics
- Need automatic heading detection based on font sizes
- Require image preservation and proper positioning
- Want a single, tested script rather than building custom logic
- Need verbose debugging information during conversion

**Comparison with Other Tools:**

| Aspect             | pdf_to_md.py           | pymupdf            | marker-pdf     |
| :----------------- | :--------------------- | :----------------- | :------------- |
| Image extraction   | Yes, with positioning  | Yes                | Yes            |
| Heading detection  | Font-size based        | Layout model       | ML-based       |
| Image organization | Separate `images/` dir | Yes                | Separate files |
| Debugging output   | Verbose logging        | Progress reporting | Limited        |
| Use case           | Production docs        | LLM input          | Academic       |

## Common Gotchas

**Scanned PDFs (of images, not text)**

- Here OCR is required to recognize the characters
- pymupdf supports OCR through its Markdown converter, provided that an OCR
  engine and appropriate language data are available
- example:

```python
import pymupdf

with pymupdf.open("scanned.pdf") as doc:
    md = doc.to_markdown(
    use_ocr = True,
    ocr_language = "eng"
    )
with open("output.md","w",encoding = "utf-8") as f:
    f.write(md)
```

- OCR quality depends on image resolution, scan quality, document layout and
  language support

## When to Use Each Library

- **Use `pymupdf`** for:
    - Speed is priority
    - Feeding PDFs to LLMs
    - Simple documents without complex layouts
    - Batch processing large numbers of PDFs

- **Use `marker-pdf`** for:
    - Academic papers and research documents
    - High-fidelity structure preservation
    - Complex layouts with tables and figures
    - When accuracy is critical

- **Use `markitdown`** for:
    - Mixed document formats (PDF, DOCX, PPTX)
    - Need consistency across formats
    - General-purpose document conversion

- **Use `pdfplumber`** for:
    - Custom extraction logic
    - Full programmatic control
    - Complex pipelines
    - When you need to build your own Markdown conversion

## Other Tools Worth Exploring

Beyond the four libraries above, three other tools are worth a look:

- [docling](https://github.com/docling-project/docling): A document conversion
  toolkit supporting advanced PDF layout analysis, tables, reading order and
  Markdown export
- [Nougat](https://github.com/facebookresearch/nougat): An OCR-based approach
  designed for academic papers. A standout feature is that it can render
  mathematical notations
- [Pandoc](https://pandoc.org/): If the original LaTeX source is available,
  convert it directly to Markdown instead of reconstructing the document from
  PDF
    - for example, to convert a LaTeX source file:

```bash
pandoc paper.tex -f latex -t gfm -o paper.md
```

## What's Next

- Converting a PDF you already have is half the job for academic papers you
  usually start from a link
- A follow-up post covers
  [download_academic_paper_to_md.py](https://github.com/causify-ai/helpers/blob/master/dev_scripts_helpers/download/download_academic_paper_to_md.py),
  which runs the whole flow in one command
