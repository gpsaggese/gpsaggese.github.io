#!/usr/bin/env -S uv run

# /// script
# dependencies = [
#   "llm",
#   "openai",
#   "pandas>=2.0.0",
#   "pdf2image",
#   "pillow",
#   "python-dotenv",
#   "pyyaml",
#   "requests",
#   "tqdm",
# ]
# ///

r"""
Generate a PDF with the lecture commentaries from slides lecture source.

This script performs multiple steps:
1. Generate PDF of the slides using `notes_to_pdf.py`
2. Extract PNG images from the generated PDF (tracked independently of step 3
   for incremental runs: e.g., deleting only the PNG dir re-extracts without
   forcing a full markdown regeneration)
3. Generate lecture commentary: pair each slide's markdown with its PNG and
   generate per-slide LLM commentary, tagging the output with the git
   hash/timestamp of generation
4. Add the generated markdown to git
5. Convert to PDF using pandoc
6. Convert to HTML using pandoc
7. Open the PDF in Skim (if `--open_pdf` is specified)
8. Open the HTML file in the default browser (if `--open_html` is specified)

# Usage Example

- Generate the lecture commentary PDF for DATA605 lesson 01.1:
> gen_lecture_commentary.py data605/01.1

- Generate the lecture commentary PDF for MSML610 lesson 02.3:
> gen_lecture_commentary.py msml610/02.3

The output looks like:
https://github.com/gpsaggese/gpsaggese.github.io/blob/master/data605/lectures_commentary
or

```
> ls -1 data605/lectures_commentary/Lesson01.1-Intro.*
data605/lectures_commentary/Lesson01.1-Intro.book_chapter.html
data605/lectures_commentary/Lesson01.1-Intro.book_chapter.md
data605/lectures_commentary/Lesson01.1-Intro.book_chapter.pdf

data605/lectures_commentary/Lesson01.1-Intro.png:
slides001.png
slides002.png
...
```

Import as:

import class_scripts.gen_lecture_commentary as clgelcom
"""

import argparse
import glob
import logging
import os
from typing import List

import pdf2image  # type: ignore
import tqdm
from PIL import ImageOps

import class_scripts.common_utils as csccouti
import class_scripts.slides_utils as cscsluti
import dev_scripts_helpers.dockerize.lib_prettier as dshdlipr
import helpers.hdbg as hdbg
import helpers.hgit as hgit
import helpers.hio as hio
import helpers.hparser as hparser
import helpers.hprint as hprint
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)

# #############################################################################
# PNG processing
# #############################################################################

# Map `--image_type` values to the corresponding PIL save format and file
# extension.
_IMAGE_TYPE_TO_PIL_INFO = {
    "png": ("PNG", "png"),
    "jpg": ("JPEG", "jpg"),
}

# Border drawn around each extracted slide image.
_IMAGE_BORDER_WIDTH_PX = 3
_IMAGE_BORDER_COLOR = "black"


def get_image_extension(image_type: str) -> str:
    """
    Get the file extension corresponding to an `--image_type` value.

    :param image_type: image type (e.g., "png", "jpg")
    :return: file extension without the leading dot (e.g., "png", "jpg")
    """
    hdbg.dassert_in(
        image_type,
        _IMAGE_TYPE_TO_PIL_INFO,
        "Invalid image type specified",
    )
    _, extension = _IMAGE_TYPE_TO_PIL_INFO[image_type]
    return extension


def _extract_png_from_pdf(
    input_pdf_file: str,
    output_png_dir: str,
    image_type: str,
    *,
    dpi: int = 300,
    add_border: bool = False,
) -> None:
    """
    Extract slide images from PDF file using pdf2image.

    :param input_pdf_file: path to input PDF file
    :param output_png_dir: directory to save the extracted images
    :param dpi: DPI resolution for output images
    :param image_type: image format to save (e.g., "png", "jpg")
    :param add_border: if True, draw a solid border around each extracted
        image (baked into the pixels, so it shows up in every output format,
        e.g., PDF and HTML)
    """
    hdbg.dassert_file_exists(input_pdf_file)
    hdbg.dassert_in(
        image_type,
        _IMAGE_TYPE_TO_PIL_INFO,
        "Invalid image type specified",
    )
    pil_format, extension = _IMAGE_TYPE_TO_PIL_INFO[image_type]
    _LOG.info("Extracting %s images from PDF: %s", image_type, input_pdf_file)
    # Create output directory.
    hio.create_dir(output_png_dir, incremental=False)
    _LOG.info("Output image directory: %s", output_png_dir)
    # Convert PDF pages to images.
    _LOG.info("Converting PDF to images with DPI=%d", dpi)
    images = pdf2image.convert_from_path(input_pdf_file, dpi=dpi)
    num_pages = len(images)
    hdbg.dassert_lt(0, num_pages, "No pages found in PDF file:", input_pdf_file)
    _LOG.info("Found %d pages in PDF", num_pages)
    # Save each page as an image file.
    for page_num, image in enumerate(
        tqdm.tqdm(images, desc="Extracting pages"), start=1
    ):
        # Format filename with zero-padded page number.
        output_filename = f"slides{page_num:03d}.{extension}"
        output_path = os.path.join(output_png_dir, output_filename)
        # JPEG has no alpha channel, so convert away from RGBA before saving.
        if pil_format == "JPEG" and image.mode != "RGB":
            image = image.convert("RGB")
        if add_border:
            image = ImageOps.expand(
                image,
                border=_IMAGE_BORDER_WIDTH_PX,
                fill=_IMAGE_BORDER_COLOR,
            )
        image.save(output_path, pil_format)
        _LOG.debug("Saved: %s", output_filename)
    _LOG.info(
        "Successfully extracted %d %s images to %s",
        num_pages,
        image_type,
        output_png_dir,
    )


def _get_png_files_from_directory(png_dir: str, image_type: str) -> List[str]:
    """
    Get sorted list of slide image files from directory.

    :param png_dir: directory containing the slide image files
    :param image_type: image format to look for (e.g., "png", "jpg")
    :return: sorted list of image file paths with pattern slides*.<extension>
    """
    hdbg.dassert_dir_exists(png_dir)
    extension = get_image_extension(image_type)
    # List all image files matching the pattern slides*.<extension>.
    png_files = []
    for filename in os.listdir(png_dir):
        if filename.startswith("slides") and filename.endswith(f".{extension}"):
            png_files.append(os.path.join(png_dir, filename))
    # Sort files to ensure correct ordering.
    png_files.sort()
    _LOG.info("Found %d image files in directory: %s", len(png_files), png_dir)
    return png_files


def _is_png_dir_populated(png_dir: str, image_type: str) -> bool:
    """
    Check whether an image directory already contains extracted slide images.

    Used to make `--no_incremental` handling for image extraction
    independent of the markdown artifact: e.g., deleting only the image dir
    triggers re-extraction without forcing a full markdown regeneration.

    :param png_dir: directory expected to contain `slides*.<extension>` files
    :param image_type: image format to look for (e.g., "png", "jpg")
    :return: True if the directory exists and contains at least one image
        file
    """
    extension = get_image_extension(image_type)
    is_populated = os.path.isdir(png_dir) and bool(
        glob.glob(os.path.join(png_dir, f"slides*.{extension}"))
    )
    return is_populated


# #############################################################################
# Commentary
# #############################################################################


# Default system prompt for the LLM, stored in a sibling file so that it can
# be edited without touching the code.
_SYSTEM_PROMPT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "prompt.generate_lecture_commentary.md",
)
_DEFAULT_SYSTEM_PROMPT = hio.from_file(_SYSTEM_PROMPT_FILE)


# Backends supported by `_generate_slide_commentary()`.
# - "hllm": `helpers.hllm.get_completion()`, supports passing the slide's
#   images as multi-modal context
# - "hllm_cli_lib": `helpers.hllm_cli.apply_llm(backend="library")`,
#   text-only (no image support)
# - "hllm_cli_exec": `helpers.hllm_cli.apply_llm(backend="executable")`,
#   text-only (no image support), shells out to simonw's `llm` CLI
#   executable
_LLM_BACKENDS = csccouti.LLM_BACKENDS


def _generate_slide_commentary(
    slide_content: str,
    system_prompt: str,
    model: str,
    llm_backend: str,
) -> str:
    """
    Generate commentary for a single slide using LLM.

    :param slide_content: markdown content of the slide
    :param system_prompt: system prompt for the LLM
    :param model: LLM model to use
    :param llm_backend: which LLM backend to use, one of `_LLM_BACKENDS`
        - "hllm": also feeds the slide's images to the LLM as multi-modal
          context
        - "hllm_cli_lib" / "hllm_cli_exec": text-only, since
          `hllm_cli.apply_llm()` has no image support
    :return: generated commentary text
    """
    hdbg.dassert_in(llm_backend, _LLM_BACKENDS)
    _LOG.debug("Generating commentary for slide")
    # Process images from slide.
    processed_slides, images_as_base64 = cscsluti.process_slide_images(
        [slide_content]
    )
    user_prompt = processed_slides[0]
    # Get completion from LLM, cached on disk since re-running this script
    # incrementally should not re-pay for an unchanged slide.
    return csccouti.call_llm_cached(
        user_prompt,
        system_prompt,
        model,
        llm_backend,
        images_as_base64=tuple(images_as_base64),
    )


def _generate_lecture_commentary(
    input_file: str,
    output_dir: str,
    input_png_dir: str,
    image_type: str,
    model: str,
    llm_backend: str,
    *,
    output_file: str = "",
    image_width: str = "80%",
    add_new_page: bool = False,
) -> None:
    r"""
    Generate book chapter from markdown slides and a directory of slide PNGs.

    :param input_file: path to input markdown file with slides
    :param output_dir: directory to save output files
    :param input_png_dir: directory containing the slide image files
        (slides*.<extension>), already extracted from the corresponding PDF
        (see `_extract_png_from_pdf()`)
    :param output_file: path to the output book chapter markdown file
        - Default: `{output_dir}/{base_name}.book_chapter.md`
    :param image_width: width of images in output (e.g., "80%", "50%")
    :param add_new_page: if True, add `\newpage` commands before each slide
    :param image_type: image format of the files in `input_png_dir` (e.g.,
        "png", "jpg")
    :param model: LLM model to use, or "" to use the backend's default
    :param llm_backend: which LLM backend to use to generate commentary, one
        of `_LLM_BACKENDS` (see `_generate_slide_commentary()`)
    """
    hdbg.dassert_file_exists(input_file)
    hdbg.dassert_dir_exists(input_png_dir)
    # Create output directory.
    hio.create_dir(output_dir, incremental=True)
    # Extract base name from input file.
    input_basename = os.path.basename(input_file)
    if input_basename.endswith(".smd"):
        base_name = input_basename[:-4]
    elif input_basename.endswith(".md"):
        base_name = input_basename[:-3]
    else:
        base_name = input_basename
    _LOG.info("Using base name: %s", base_name)
    _LOG.info("Reading slides from: %s", input_file)
    # Extract title from markdown file for YAML preamble.
    title = csccouti.extract_title_from_markdown(input_file)
    # Extract slides from markdown file.
    slides, titles = cscsluti.extract_slides_from_file(input_file)
    num_slides = len(slides)
    _LOG.info("Found %d slides in markdown file", num_slides)
    # Get PNG files from directory.
    png_files = _get_png_files_from_directory(input_png_dir, image_type)
    num_pngs = len(png_files)
    _LOG.info("Found %d PNG files in directory", num_pngs)
    # Check that slide count matches PNG count.
    hdbg.dassert_eq(
        # +1 because the first slide is the title slide.
        num_slides + 1,
        num_pngs,
        "Number of slides in markdown (%d) does not match number of PNG files (%d)",
        num_slides,
        num_pngs,
    )
    # Generate commentary for each slide.
    output_parts = []
    # Add YAML preamble with title if available.
    if title:
        yaml_preamble = f'---\ntitle: "{title}"\n---\n'
        output_parts.append(yaml_preamble)
    # Add a provenance tag with the git hash and timestamp of generation, so
    # that we can tell from which commit and when this file was generated.
    generation_tag = hgit.get_generation_tag()
    output_parts.append(f"<!-- {generation_tag} -->\n")
    # First, handle the title slide (first PNG, no content).
    _LOG.info("Processing title slide (1/%d)", num_slides + 1)
    slide_output = []
    if add_new_page:
        slide_output.append("\\newpage")
        slide_output.append("")
    # Add centered image with specified width and empty alt text.
    # The `<center>` tag centers the image in the HTML output, but pandoc
    # drops raw HTML when rendering to PDF/LaTeX, so we also wrap the image
    # in a raw LaTeX `center` environment (via `{=latex}` raw blocks) to
    # center it in the PDF output too.
    slide_output.append(
        hprint.dedent(
            rf"""
            <center>
            ```{{=latex}}
            \begin{{center}}
            ```
            ![]({png_files[0]}){{width={image_width}}}
            ```{{=latex}}
            \end{{center}}
            ```
            </center>
            """
        )
    )
    output_parts.append("\n".join(slide_output))
    # Then process content slides (slides from markdown with corresponding PNGs).
    # Note: png_files[0] is the title slide, so we pair slides[i] with png_files[i+1].
    for idx, (slide_content, slide_title, png_path) in enumerate(
        tqdm.tqdm(
            zip(slides, titles, png_files[1:]),
            total=num_slides,
            desc="Processing slides",
        ),
        start=2,
    ):
        _LOG.info("Processing slide %d/%d", idx, num_slides + 1)
        # Create output for this slide.
        slide_output = []
        # Add page break before slide.
        if add_new_page:
            slide_output.append("\\newpage")
            slide_output.append("")
        # Add title, image, and commentary.
        # Use original slide title from input markdown with idx/tot format.
        full_title = f"{idx} / {num_slides + 1}: {slide_title}"
        slide_output.append(
            hprint.dedent(
                f"""
                <center>
                # {full_title}
                </center>
                """
            )
        )
        # Add a blank line to separate the title and image `<center>` blocks.
        slide_output.append("")
        # Add centered image with specified width and empty alt text.
        # The `<center>` tag centers the image in the HTML output, but
        # pandoc drops raw HTML when rendering to PDF/LaTeX, so we also
        # wrap the image in a raw LaTeX `center` environment (via
        # `{=latex}` raw blocks) to center it in the PDF output too.
        slide_output.append(
            hprint.dedent(
                rf"""
                <center>
                ```{{=latex}}
                \begin{{center}}
                ```
                ![]({png_path}){{width={image_width}}}
                ```{{=latex}}
                \end{{center}}
                ```
                </center>
                """
            )
        )
        # Generate commentary for this slide.
        commentary = _generate_slide_commentary(
            slide_content, _DEFAULT_SYSTEM_PROMPT, model, llm_backend
        )
        slide_output.append(commentary)
        slide_output.append("")
        # Add to output parts.
        output_parts.append("\n".join(slide_output))
    # Combine all slides.
    full_output = "\n".join(output_parts)
    # Format output with prettier.
    _LOG.info("Formatting output with prettier")
    full_output = dshdlipr.prettier_on_str(full_output, "md")
    # Write output file.
    if not output_file:
        output_file = os.path.join(output_dir, f"{base_name}.book_chapter.md")
    _LOG.info("Writing output to: %s", output_file)
    hio.to_file(output_file, full_output)
    _LOG.info("Book chapter generation completed")


# #############################################################################
# CLI
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Lecture specification: 'data605/08.1', 'msml610/08.1', "
        "or file path 'msml610/lectures_source/Lesson10.2-Name.smd'",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print the commands that would be executed without running them",
    )
    parser.add_argument(
        "--no_incremental",
        action="store_true",
        help=(
            "Force regeneration of intermediate files even if they already "
            "exist (by default, steps are skipped if their output already "
            "exists)"
        ),
    )
    parser.add_argument(
        "--image_type",
        type=str,
        choices=["png", "jpg"],
        default="png",
        help="Image format to extract slides as",
    )
    parser.add_argument(
        "--llm_backend",
        type=str,
        choices=_LLM_BACKENDS,
        default="hllm",
        help=(
            "LLM backend to use for slide commentary generation: 'hllm' "
            "(default) feeds the slide's images to the LLM as multi-modal "
            "context, 'hllm_cli_lib' (library) and 'hllm_cli_exec' "
            "(simonw's `llm` CLI executable) are text-only"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="LLM model to use (e.g., 'gpt-4o', 'claude-opus-4'); empty "
        "string (default) uses the --llm_backend's default model",
    )
    parser.add_argument(
        "--open_pdf",
        action="store_true",
        help="Open the generated PDF in Skim",
    )
    parser.add_argument(
        "--open_html",
        action="store_true",
        help="Open the generated HTML file in the default browser",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Parse and validate arguments.
    dir_arg, lesson_arg = csccouti.parse_lesson_spec(args.input)
    csccouti.validate_dir_lesson_args(dir_arg, lesson_arg)
    # Get source name.
    src_name = csccouti.get_source_name(dir_arg, lesson_arg)
    input_file = f"{dir_arg}/lectures_source/{src_name}"
    # Precompute the paths of all intermediate/output files, so that we can
    # skip steps whose output already exists (unless --no_incremental).
    dst_name = csccouti.get_output_name(src_name, ".pdf")
    tmp_pdf = f"tmp.{dst_name}"
    out_dir = f"{dir_arg}/lectures_commentary"
    basename = os.path.splitext(src_name)[0]
    image_extension = get_image_extension(args.image_type)
    png_dir = f"{out_dir}/{basename}.{image_extension}"
    book_chapter_md = f"{out_dir}/{basename}.book_chapter.md"
    pdf_file_name = f"{out_dir}/{basename}.book_chapter.pdf"
    html_file_name = f"{out_dir}/{basename}.book_chapter.html"
    do_incremental = not args.no_incremental
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Step 1: Generate the PDF.
    if do_incremental and os.path.exists(tmp_pdf):
        _LOG.warning("Step 1: Skipping, '%s' already exists", tmp_pdf)
    else:
        _LOG.info("Step 1: Generating PDF from lecture source")
        cmd = (
            "notes_to_pdf.py "
            f"--input {input_file} "
            f"--output {tmp_pdf} "
            "--type slides --toc_type remove_headers"
        )
        hsystem.system(cmd, print_command=True, dry_run=args.dry_run)
    # Step 2: Extract slide images from the PDF.
    # This is tracked independently from the markdown artifact (Step 3) so
    # that deleting only the image dir triggers re-extraction without forcing
    # a full markdown/commentary regeneration.
    csccouti.ensure_dir_exists(out_dir)
    if do_incremental and _is_png_dir_populated(png_dir, args.image_type):
        _LOG.warning("Step 2: Skipping, '%s' already populated", png_dir)
    else:
        _LOG.info("Step 2: Extracting %s images from PDF", args.image_type)
        if args.dry_run:
            _LOG.warning(
                "As per user request, not extracting images for '%s'",
                tmp_pdf,
            )
        else:
            _extract_png_from_pdf(
                tmp_pdf,
                png_dir,
                args.image_type,
                dpi=200,
                add_border=True,
            )
    # Step 3: Generate book chapter.
    if do_incremental and os.path.exists(book_chapter_md):
        _LOG.warning("Step 3: Skipping, '%s' already exists", book_chapter_md)
    else:
        _LOG.info("Step 3: Generating book chapter")
        if args.dry_run:
            _LOG.warning(
                "As per user request, not generating book chapter for '%s'",
                input_file,
            )
        else:
            _generate_lecture_commentary(
                input_file,
                out_dir,
                png_dir,
                args.image_type,
                args.model,
                args.llm_backend,
                output_file=book_chapter_md,
            )
    # Step 4: Track the generated markdown file in git.
    _LOG.info("Step 4: Adding book chapter markdown to git")
    csccouti.git_add_with_retry(book_chapter_md, dry_run=args.dry_run)
    # Step 5: Convert to PDF using pandoc.
    if do_incremental and os.path.exists(pdf_file_name):
        _LOG.warning("Step 5: Skipping, '%s' already exists", pdf_file_name)
    else:
        _LOG.info("Step 5: Converting to PDF using pandoc")
        csccouti.convert_markdown_to_pdf(
            book_chapter_md, pdf_file_name, script_dir, dry_run=args.dry_run
        )
    # Step 6: Convert to HTML using pandoc.
    if do_incremental and os.path.exists(html_file_name):
        _LOG.warning("Step 6: Skipping, '%s' already exists", html_file_name)
    else:
        _LOG.info("Step 6: Converting to HTML using pandoc")
        cmd = (
            f"pandoc {book_chapter_md} -o {html_file_name} "
            f"--standalone "
            # Inline every locally-referenced image (and other resources) as
            # a base64 data URI, so the HTML is self-contained and portable
            # regardless of where it is opened from.
            f"--embed-resources "
            f"--css={script_dir}/book-style.css "
            f"--highlight-style=tango"
        )
        hsystem.system(cmd, print_command=True, dry_run=args.dry_run)
    _LOG.info("PDF file: %s", pdf_file_name)
    _LOG.info("HTML file: %s", html_file_name)
    # Step 7: Open the PDF in Skim.
    if args.open_pdf:
        _LOG.info("Step 7: Opening PDF in Skim")
        cmd = f"open -a /Applications/Skim.app {pdf_file_name}"
        hsystem.system(cmd, print_command=True, dry_run=args.dry_run)
    # Step 8: Open the HTML file in the default browser.
    if args.open_html:
        _LOG.info("Step 8: Opening HTML file in default browser")
        cmd = f"open {html_file_name}"
        hsystem.system(cmd, print_command=True, dry_run=args.dry_run)
    _LOG.info("Book chapter generated: %s", pdf_file_name)


if __name__ == "__main__":
    _main(_parse())
