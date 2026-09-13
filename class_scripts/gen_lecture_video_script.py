#!/usr/bin/env python

"""
Generate lecture script from slides.

This script performs multiple steps:
1. Generate per-slide-group script by sending groups of slides to an LLM
2. Generate intro using llm_cli.py
3. Generate outro using llm_cli.py
4. Combine intro, script, and outro
5. Lint the final script

# Usage Example

- Generate the lecture video script for DATA605 lesson 01.1:
> gen_lecture_video_script.py data605/01.1

- Generate the lecture video script for MSML610 lesson 02.3:
> gen_lecture_video_script.py msml610/02.3

- Generate script with a larger slide grouping:
> gen_lecture_video_script.py msml610/02.3 --slides_per_group 5

- Process only a range of slides:
> gen_lecture_video_script.py data605/01.1 --limit 1:5

Import as:

import class_scripts.gen_lecture_video_script as clgelesc
"""

import argparse
import logging
from typing import List, Tuple

import tqdm

import class_scripts.common_utils as csccouti
import class_scripts.slides_utils as cscsluti
import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hllm as hllm
import helpers.hparser as hparser
import helpers.hselect_input_output as hseinout
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################


# Prompt for generating intro.
INTRO_PROMPT = (
    "You are a college professor and you need to do an introduction in 50 word "
    "the content of the slides starting with In this lesson we will discuss"
)

# Prompt for generating outro.
OUTRO_PROMPT = (
    "You are a college professor and you need to summarize what was discussed "
    "in less than 50 word in the slides like In this lesson we have discussed"
)

# Default system prompt used to turn one group of slides into a spoken-
# discussion script.
_SLIDE_SCRIPT_SYSTEM_PROMPT = """
You are a college professor expert of machine learning and big data.

Given the following slides in markdown format create a discussion of the slide
to highlight the most important points of each slide
- Use plain language and do not use fancy words
- Create bullet points for the discussion following the same structure as the
  original slide
- The discussion for each slide should contain around 100 words
- Do not use bold or italicize the text
- Create a short transitions in less than 20 words between slides when needed.
- Use "we" and "let's" instead of saying "This slide says"

The output should have a format like:

# <Title>

Description of the slide
"""


# #############################################################################
# Slide script generation
# #############################################################################


def _process_slides_group(
    slides_group: List[str],
    system_prompt: str,
    model: str,
) -> str:
    """
    Process a group of slides through LLM to generate presentation script.

    :param slides_group: list of slide contents to process
    :param system_prompt: system prompt for the LLM
    :param model: LLM model to use
    :return: generated script content
    """
    hdbg.dassert_isinstance(slides_group, list)
    hdbg.dassert_lt(0, len(slides_group))
    # Process images from slides.
    processed_slides, images_as_base64 = cscsluti.process_slide_images(
        slides_group
    )
    # Combine slides into user prompt.
    user_prompt = "\n\n".join(processed_slides)
    _LOG.debug("Processing %d slides with LLM", len(processed_slides))
    if images_as_base64:
        _LOG.info("Including %d images in LLM request", len(images_as_base64))
    # Get completion from LLM with images if present.
    response = hllm.get_completion(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        model=model,
        cache_mode="NORMAL",
        temperature=0.1,
        # images_as_base64=tuple(images_as_base64) if images_as_base64 else None,
    )
    hdbg.dassert_isinstance(response, str)
    return response


def generate_slide_script(
    in_file: str,
    out_file: str,
    *,
    slides_per_group: int = 3,
    limit_range: Tuple[int, int] = (0, 0),
) -> None:
    """
    Generate presentation script from markdown slides.

    Groups slides (identified by headers starting with '*') and sends each
    group to an LLM to produce a spoken-discussion script.

    :param in_file: path to input markdown file
    :param out_file: path to output script file
    :param slides_per_group: number of slides to process in each LLM
        call
    :param limit_range: 0-indexed inclusive `(start, end)` slide range to
        process; `(0, 0)` means process all slides
    """
    _LOG.info("Reading slides from: %s", in_file)
    slides, _ = cscsluti.extract_slides_from_file(in_file)
    _LOG.info("Found %d slides total", len(slides))
    # Apply limit range if specified.
    if limit_range != (0, 0):
        start, end = limit_range
        slides = slides[start : end + 1]
        _LOG.info("Limited to slides %d-%d (%d slides)", start, end, len(slides))
    # Process slides in groups.
    output_parts = []
    total_groups = (len(slides) + slides_per_group - 1) // slides_per_group
    for i in tqdm.tqdm(
        range(0, len(slides), slides_per_group),
        total=total_groups,
        desc="Processing slide groups",
    ):
        group_end = min(i + slides_per_group, len(slides))
        slides_group = slides[i:group_end]
        _LOG.info("Processing slides %d-%d", i + 1, group_end)
        # Process the group.
        script_content = _process_slides_group(
            slides_group=slides_group,
            system_prompt=_SLIDE_SCRIPT_SYSTEM_PROMPT,
            model="",
        )
        output_parts.append(script_content)
    # Combine all generated scripts.
    full_script = "\n\n".join(output_parts)
    # Write output.
    _LOG.info("Writing script to: %s", out_file)
    hio.to_file(out_file, full_script)
    _LOG.info("Script generation completed")


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
        "--slides_per_group",
        action="store",
        type=int,
        default=3,
        help="Number of slides to process per LLM call",
    )
    hseinout.add_limit_range_arg(parser)
    hparser.add_verbosity_arg(parser)
    return parser


def generate_lecture_video_script(
    input_file: str,
    output_file: str,
    *,
    slides_per_group: int = 3,
    limit_range: Tuple[int, int] = (0, 0),
) -> None:
    """
    Generate a complete lecture video script from a slides source file.

    Runs the full pipeline: per-slide-group script, intro, outro,
    combine, and lint. This is the function backing this script's CLI, and
    is also called directly by callers (e.g. `for_loop_lessons.py`) that
    already have resolved `input_file`/`output_file` paths.

    :param input_file: path to the lecture source file (e.g., a `.smd`
        file)
    :param output_file: path to write the final combined, linted script to
    :param slides_per_group: number of slides to process in each LLM call
    :param limit_range: 0-indexed inclusive `(start, end)` slide range to
        process; `(0, 0)` means process all slides
    """
    # Step 1: Generate script.
    _LOG.info("Step 1: Generating per-slide-group script")
    generate_slide_script(
        input_file,
        output_file,
        slides_per_group=slides_per_group,
        limit_range=limit_range,
    )
    # Step 2: Generate intro.
    _LOG.info("Step 2: Generating intro")
    intro_file = "tmp.gen_lecture_video_script.intro.txt"
    cmd = f"llm_cli.py -i {output_file} -p '{INTRO_PROMPT}' -o {intro_file}"
    hsystem.system(cmd)
    # Step 3: Generate outro.
    _LOG.info("Step 3: Generating outro")
    outro_file = "tmp.gen_lecture_video_script.outro.txt"
    cmd = f"llm_cli.py -i {output_file} -p '{OUTRO_PROMPT}' -o {outro_file}"
    hsystem.system(cmd)
    # Step 4: Combine intro, script, and outro.
    _LOG.info("Step 4: Combining intro, script, and outro")
    intro_text = hio.from_file(intro_file)
    script_text = hio.from_file(output_file)
    outro_text = hio.from_file(outro_file)
    # Build combined content.
    combined_parts = [
        "# Intro",
        intro_text.strip(),
        "",
        script_text.strip(),
        "",
        "# Outro",
        outro_text.strip(),
    ]
    combined_text = "\n".join(combined_parts)
    # Write to temporary file.
    tmp_file = "tmp.gen_lecture_video_script.combined.txt"
    hio.to_file(tmp_file, combined_text)
    # Move to final location.
    hio.to_file(output_file, combined_text)
    # Step 5: Lint the final script.
    _LOG.info("Step 5: Linting the final script")
    cmd = (
        f"lint_text.py -i {output_file} -o {output_file} "
        f"--use_dockerized_prettier --action prettier --action frame_chapters"
    )
    hsystem.system(cmd)
    _LOG.info("Lecture script generated: %s", output_file)


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Parse and validate arguments.
    dir_arg, lesson_arg = csccouti.parse_lesson_spec(args.input)
    csccouti.validate_dir_lesson_args(dir_arg, lesson_arg)
    # Get source and destination names.
    src_name = csccouti.get_source_name(dir_arg, lesson_arg)
    dst_name = csccouti.get_output_name(src_name, ".script.txt")
    # Build paths.
    input_file = f"{dir_arg}/lectures_source/{src_name}"
    output_dir = f"{dir_arg}/lectures_video_script"
    output_file = f"{output_dir}/{dst_name}"
    # Ensure output directory exists.
    csccouti.ensure_dir_exists(output_dir)
    # Parse limit range.
    limit_range = hseinout.parse_limit_range_args(args) or (0, 0)
    # Run the full pipeline.
    generate_lecture_video_script(
        input_file,
        output_file,
        slides_per_group=args.slides_per_group,
        limit_range=limit_range,
    )


if __name__ == "__main__":
    _main(_parse())
