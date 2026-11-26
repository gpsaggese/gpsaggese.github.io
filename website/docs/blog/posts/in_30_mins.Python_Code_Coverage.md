---
title: "Python Code Coverage In 30 Mins"
draft: false
authors:
    - gpsaggese
date: 2026-04-17
description:
categories:
    - Python
---

TL;DR: Python code coverage measures how much of your source code is tested,
helping you improve test quality.

<!-- more -->

# Python Code Coverage

## Overview

- Code coverage is a metric used to measure how much of your source code is
  exercised during test execution
- In Python, combining `pytest` (a testing framework) with the `coverage` module
  allows you to assess and improve the quality of your tests

## Tools Required

- **pytest**: A framework for writing and running Python tests.
- **coverage.py**: A tool to measure code coverage of Python programs.

- Install both using pip:
    ```bash
    > pip install pytest coverage pytest-cov
    ```

# Basic Workflow

## Text report

- Assume you have a simple module `my_module.py` and a corresponding
  `test_math.py` containing unit tests:
    ```python
    # my_module.py
    def add(a, b):
        return a + b
    ```

    ```python
    # test_math.py
    from my_module import add

    def test_add():
        assert add(2, 3) == 5
    ```

- Run tests with coverage and collects coverage data:

    ```bash
    > pytest --cov=test_math.py
    ```

- Once the data is collected with `pytest`, it can be post-processed in several
  ways using `coverage`

- E.g., display a coverage summary in the terminal:

    ```bash
    > coverage report

    Name           Stmts   Miss  Cover
    ----------------------------------
    my_module.py      10      2    80%
    ```

- The table shows the coverage data:
    - `Stmts` is the number of statements
    - `Miss` shows how many statements were not executed
    - `Cover` is the percentage of statements covered by tests

## Interactive Coverage Analysis

- To view a detailed coverage report in your browser:

    ```bash
    > coverage html
    ```

- This generates an `htmlcov` directory containing an `index.html` file you can
  open in a browser

## Options of `pytest` and `coverage`

- The options for `pytest coverage` are:
    ```verbatim
    coverage reporting with distributed testing support:
      --cov=[SOURCE]        Path or package name to measure during execution (multi-
                            allowed). Use --cov= to not do any source filtering and
                            record everything.
      --cov-reset           Reset cov sources accumulated in options so far.
      --cov-report=TYPE     Type of report to generate: term, term-missing,
                            annotate, html, xml, json, lcov (multi-allowed). term,
                            term-missing may be followed by ":skip-covered".
                            annotate, html, xml, json and lcov may be followed by
                            ":DEST" where DEST specifies the output location. Use
                            --cov-report= to not generate any output.
      --cov-config=PATH     Config file for coverage. Default: .coveragerc
      --no-cov-on-fail      Do not report coverage if test run fails. Default: False
      --no-cov              Disable coverage report completely (useful for
                            debuggers). Default: False
      --cov-fail-under=MIN  Fail if the total coverage is less than MIN.
      --cov-append          Do not delete coverage but append to current. Default:
                            False
      --cov-branch          Enable branch coverage.
      --cov-precision=COV_PRECISION
                            Override the reporting precision.
      --cov-context=CONTEXT
                            Dynamic contexts to use. "test" for now.
    ```

- The `coverage` command has the following options:

    ```bash
    > coverage --help
    Coverage.py, version 7.8.0 with C extension
    Measure, collect, and report on code coverage in Python programs.

    usage: coverage <command> [options] [args]

    Commands:
          annotate    Annotate source files with execution information.
          combine     Combine a number of data files.
          debug       Display information about the internals of coverage.py
          erase       Erase previously collected coverage data.
          help        Get help on using coverage.py.
          html        Create an HTML report.
          json        Create a JSON report of coverage results.
          lcov        Create an LCOV report of coverage results.
          report      Report coverage stats on modules.
          run         Run a Python program and measure code execution.
          xml         Create an XML report of coverage results.

    Use "coverage help <command>" for detailed help on any command.
    Full documentation is at https://coverage.readthedocs.io/en/7.8.0
    ```

# Workflows

- Let's use the repo `//helpers` as an example

## Generate Coverage Data from Unit tests

- You want to measure the coverage of the codebase from a set of unit tests
  (e.g., all the tests `helpers/test/test_hmarkdown*.py`)

    ```bash
    > pytest helpers/test/test_hmarkdown*.py --cov 2>&1 | tee log.txt
    ...
    collected 117 items

    helpers/test/test_hmarkdown.py::Test_header_list_to_vim_cfile1::test_get_header_list1 (0.00 s) PASSED [  0%]
    helpers/test/test_hmarkdown.py::Test_header_list_to_markdown1::test_mode_headers1 (0.00 s) PASSED [  1%]
    helpers/test/test_hmarkdown.py::Test_header_list_to_markdown1::test_mode_list1 (0.00 s) PASSED [  2%]
    helpers/test/test_hmarkdown.py::Test_replace_fenced_blocks_with_tags1::test1 (0.00 s) PASSED [  3%]
    ...

    ================================ tests coverage ================================
    _______________ coverage: platform linux, python 3.12.3-final-0 ________________

    Name                                     Stmts   Miss Branch BrPart  Cover
    --------------------------------------------------------------------------
    __init__.py                                  0      0      0      0   100%
    conftest.py                                 79     38     18      7    49%
    helpers/__init__.py                          0      0      0      0   100%
    helpers/hcoverage.py                        69     53     14      0    19%
    helpers/hdbg.py                            394    267    136     23    29%
    helpers/hdocker.py                         564    510    102      0     8%
    helpers/henv.py                            216     92     46      4    50%
    helpers/hgit.py                            573    491    130      0    12%
    helpers/hintrospection.py                  125    102     48      0    13%
    helpers/hio.py                             345    244    118     12    25%
    ...
    helpers/test/test_hmarkdown_bullets.py     132      0      2      0   100%
    --------------------------------------------------------------------------
    TOTAL                                     6956   3824   1700    168    41%
    ```

- The full report shows coverage across the entire project, including both source
  and test files
// TODO(ai_gp): Convert this into a table
    - `Stmts` TODO
    - `Miss` TODO
    - `BrPart` indicates partially-covered branches
    - `Branch` shows total branches
    - `Cover` TODO

- These columns help identify conditional logic that was not fully exercised

- In general the quality of the coverage should be evaluated only running the
  entire suite of tests (including `fast`, `slow`, `superslow`)
- A good approximation is to judge the coverage for the target unit tests
    - E.g., for `dev_scripts_helpers/documentation`, you can run the tests
      under `dev_scripts_helpers/documentation/test`

pytest dev_scripts_helpers/documentation/test --cov
coverage report --include=dev_scripts_helpers/documentation/test/*.py

## Post-processing Coverage Data for Target Files

- Post-process and generate the coverage report for only certain target code
  (e.g., `helpers/hmarkdown*.py`)

    ```bash
    > coverage report --include=helpers/hmarkdown*.py
    Name                                 Stmts   Miss Branch BrPart  Cover
    ----------------------------------------------------------------------
    helpers/hmarkdown.py                     9      0      0      0   100%
    helpers/hmarkdown_bullets.py            91     24     42      5    68%
    helpers/hmarkdown_coloring.py           60     19     16      2    62%
    helpers/hmarkdown_comments.py           28     12     10      4    53%
    helpers/hmarkdown_fenced_blocks.py      55      0     14      0   100%
    helpers/hmarkdown_filtering.py          53     43      6      0    17%
    helpers/hmarkdown_formatting.py        101     34     18      1    69%
    helpers/hmarkdown_headers.py           226      6     88      7    96%
    helpers/hmarkdown_rules.py             101     19     42      3    80%
    helpers/hmarkdown_slides.py             52      2     16      4    91%
    ----------------------------------------------------------------------
    TOTAL                                  776    159    252     26    78%
    ```

- Generate the coverage sorted by % covered

    ```bash
    > coverage report --include=helpers/hmarkdown*.py --sort=cover
    Name                                 Stmts   Miss Branch BrPart  Cover
    ----------------------------------------------------------------------
    helpers/hmarkdown_filtering.py          53     43      6      0    17%
    helpers/hmarkdown_comments.py           28     12     10      4    53%
    helpers/hmarkdown_coloring.py           60     19     16      2    62%
    ...
    helpers/hmarkdown.py                     9      0      0      0   100%
    helpers/hmarkdown_fenced_blocks.py      55      0     14      0   100%
    ----------------------------------------------------------------------
    TOTAL                                  776    159    252     26    78%
    ```

## Interactive Analysis

- Analyze the coverage through browser:

    ```
    > coverage html --include=helpers/hmarkdown*.py
    Wrote HTML report to htmlcov/index.html

    > open htmlcov/index.html
    ```

// TODO(ai_gp): This seems to be malformed

- You can analyze coverage by
  Files Functions Classes

    ```
          Statements	 	Branches	 	Total
    File	 	coverage	statements	missing	excluded	 	coverage	branches	partial	 	coverage
    helpers / hmarkdown_bullets.py	 	83%	93	16	0	 	68%	44	6	 	78%
    helpers / hmarkdown_coloring.py	 	83%	86	15	0	 	64%	28	6	 	78%
    helpers / hmarkdown_comments.py	 	57%	28	12	0	 	40%	10	4	 	53%
    ```


- The columns in the interactive view:
    - **Statements coverage**: percentage of statements executed
    - **statements**: total statements
    - **missing**: statements not covered
    - **excluded**: statements excluded from coverage
    - **Branches coverage**: percentage of branches taken
    - **branches**: total branches
    - **partial**: branches partially covered
    - **Total coverage**: combined statements and branches coverage

- You can click on each column to sort by that column
- You can click on a file to see which lines are covered

## Gotchas

pytest dev_scripts_helpers/documentation/test --cov

When I run coverage I want to get a 0% if a file is not touched

> coverage report --include=dev_scripts_helpers/documentation/*.py --sort=cover
Name                                                              Stmts   Miss Branch BrPart  Cover
---------------------------------------------------------------------------------------------------
dev_scripts_helpers/documentation/piper_markdown_reader.py          329    275    110      1    13%
dev_scripts_helpers/documentation/generate_images.py                179    129     70      3    27%
dev_scripts_helpers/documentation/open_md.py                        129     92     36      2    28%
dev_scripts_helpers/documentation/summarize_md.py                   225    151     80      4    30%
dev_scripts_helpers/documentation/summarize_chapters.py              57     36     10      1    36%
dev_scripts_helpers/documentation/render_images.py                  381    174    142      9    53%
dev_scripts_helpers/documentation/convert_table.py                  111     36     30      6    62%
dev_scripts_helpers/documentation/check_links.py                    146     48     56      9    65%
dev_scripts_helpers/documentation/extract_chapters_from_text.py      99     27     34      1    74%
dev_scripts_helpers/documentation/preprocess_notes.py               349     78    140     19    77%
dev_scripts_helpers/documentation/lint_text.py                       398     71    158     18    80%
dev_scripts_helpers/documentation/documentation_utils.py            169      5     28      5    95%
dev_scripts_helpers/documentation/__init__.py                         0      0      0      0   100%
---------------------------------------------------------------------------------------------------
TOTAL                                                              2572   1122    894     78    55%

> ls -1 dev_scripts_helpers/documentation/*.py
dev_scripts_helpers/documentation/__init__.py
dev_scripts_helpers/documentation/check_ai_slop.py
dev_scripts_helpers/documentation/check_links.py
dev_scripts_helpers/documentation/clean_markdown.py
dev_scripts_helpers/documentation/concatenate_pdfs.py
dev_scripts_helpers/documentation/convert_docx_to_md.py
dev_scripts_helpers/documentation/convert_epub_to_md.py
dev_scripts_helpers/documentation/convert_pdf_to_md.py
dev_scripts_helpers/documentation/convert_png_dir_to_movie.py
dev_scripts_helpers/documentation/convert_table.py
dev_scripts_helpers/documentation/count_words.py
dev_scripts_helpers/documentation/dockerized_svg_with_inkscape.py
dev_scripts_helpers/documentation/dockerized_svg_with_rsvg_convert.py
dev_scripts_helpers/documentation/documentation_utils.py
dev_scripts_helpers/documentation/extract_chapters_from_text.py
dev_scripts_helpers/documentation/extract_from_md.py
dev_scripts_helpers/documentation/extract_gdoc_map.py
dev_scripts_helpers/documentation/extract_toc_from_txt.py
dev_scripts_helpers/documentation/generate_images.py
dev_scripts_helpers/documentation/generate_images.README.py
dev_scripts_helpers/documentation/generate_latex_sty.py
dev_scripts_helpers/documentation/generate_script_catalog.py
dev_scripts_helpers/documentation/lint_text.py
dev_scripts_helpers/documentation/notes_to_pdf.py
dev_scripts_helpers/documentation/open_md.py
dev_scripts_helpers/documentation/piper_markdown_reader.py
dev_scripts_helpers/documentation/preprocess_notes.py
dev_scripts_helpers/documentation/publish_notes.py
dev_scripts_helpers/documentation/render_images.py
dev_scripts_helpers/documentation/replace_latex.py
dev_scripts_helpers/documentation/run_pandoc.py
dev_scripts_helpers/documentation/standardize_book_filename.py
dev_scripts_helpers/documentation/summarize_chapters.py
dev_scripts_helpers/documentation/summarize_md.py
dev_scripts_helpers/documentation/transform_notes.py
dev_scripts_helpers/documentation/update_md.py

You need to do
pytest dev_scripts_helpers/documentation/test --cov=dev_scripts_helpers/documentation


2:56 PM
The issue is that coverage report only shows files that were actually imported/executed during the test run and recorded in the .coverage data file. Files that were never touched at all (no import, no execution) don't show up as 0% — they're just absent from the report entirely.

To force coverage to include untouched files as 0%, you need coverage run to know about them at collection time, via the --source option (not just --include at report time). --include/--omit only filter what's already in the data; --source tells coverage what the universe of files should be, so anything never imported still gets reported at 0%.


> coverage report --include=dev_scripts_helpers/documentation/*.py --sort=cover
Name                                                                    Stmts   Miss Branch BrPart  Cover
---------------------------------------------------------------------------------------------------------
dev_scripts_helpers/documentation/check_ai_slop.py                        151    151     28      0     0%
dev_scripts_helpers/documentation/clean_markdown.py                        25     25      2      0     0%
dev_scripts_helpers/documentation/concatenate_pdfs.py                      68     68     16      0     0%
dev_scripts_helpers/documentation/convert_docx_to_md.py                    72     72     22      0     0%
dev_scripts_helpers/documentation/convert_epub_to_md.py                   114    114     32      0     0%
dev_scripts_helpers/documentation/convert_pdf_to_md.py                    256    256    110      0     0%
dev_scripts_helpers/documentation/convert_png_dir_to_movie.py              92     92     16      0     0%
dev_scripts_helpers/documentation/count_words.py                           65     65     12      0     0%
dev_scripts_helpers/documentation/dockerized_svg_with_inkscape.py          25     25      4      0     0%
dev_scripts_helpers/documentation/dockerized_svg_with_rsvg_convert.py      25     25      4      0     0%
dev_scripts_helpers/documentation/extract_from_md.py                       32     32      2      0     0%
dev_scripts_helpers/documentation/extract_gdoc_map.py                      89     89     22      0     0%
dev_scripts_helpers/documentation/extract_toc_from_txt.py                 137    137     48      0     0%
dev_scripts_helpers/documentation/generate_latex_sty.py                   102    102     32      0     0%
dev_scripts_helpers/documentation/generate_script_catalog.py               75     75     24      0     0%
dev_scripts_helpers/documentation/notes_to_pdf.py                         494    494    128      0     0%
dev_scripts_helpers/documentation/publish_notes.py                         88     88     16      0     0%
dev_scripts_helpers/documentation/replace_latex.py                         58     58     22      0     0%
dev_scripts_helpers/documentation/run_pandoc.py                            26     26      4      0     0%
dev_scripts_helpers/documentation/standardize_book_filename.py             38     38      6      0     0%
dev_scripts_helpers/documentation/transform_notes.py                       76     76     26      0     0%
dev_scripts_helpers/documentation/update_md.py                            173    173     42      0     0%
dev_scripts_helpers/documentation/piper_markdown_reader.py                329    275    110      1    13%
dev_scripts_helpers/documentation/generate_images.py                      179    129     70      3    27%
dev_scripts_helpers/documentation/open_md.py                              129     92     36      2    28%
dev_scripts_helpers/documentation/summarize_md.py                         225    151     80      4    30%
dev_scripts_helpers/documentation/summarize_chapters.py                    57     36     10      1    36%
dev_scripts_helpers/documentation/render_images.py                        381    174    142      9    53%
dev_scripts_helpers/documentation/convert_table.py                        111     36     30      6    62%
dev_scripts_helpers/documentation/check_links.py                          146     48     56      9    65%
dev_scripts_helpers/documentation/extract_chapters_from_text.py            99     27     34      1    74%
dev_scripts_helpers/documentation/preprocess_notes.py                     349     78    140     19    77%
dev_scripts_helpers/documentation/lint_text.py                             398     71    158     18    80%
dev_scripts_helpers/documentation/documentation_utils.py                  169      5     28      5    95%
dev_scripts_helpers/documentation/__init__.py                               0      0      0      0   100%
---------------------------------------------------------------------------------------------------------
TOTAL                                                                    4853   3403   1512     78    30%

# 

- Find the changed Python files in the branch
    ```
    > i git_files --branch --file-types "py"
    18:26:55 - INFO  hdbg.py init_logger:1088                               > cmd='/Users/saggese/src/venv/client_venv.helpers/bin/invoke git_files --branch --file-types py'
    # git_files: files='', from_file='', modified=False, branch=True, last_commit=False, all_files=False, file_types='py', skip_file_types='', pbcopy=False, only_print_files=False, on_one_line=False, mode='files'
    18:26:55 - WARN  hsystem.py remove_dirs:496                             Removed dirs: helpers_root
    ================================================================================
    Results
    ================================================================================
    class_scripts/common_utils.py
    class_scripts/for_loop_lessons.py
    class_scripts/gen_slides.py
    class_scripts/gen_slides_test_utils.py
    class_scripts/generate_lecture_commentary.py
    class_scripts/generate_slide_script.py
    class_scripts/test/test_for_loop_lessons.py
    class_scripts/test/test_for_loop_slides.py
    msml610/test/test_gen_slides.py
    tutorials/lime/01.API.lime.py
    tutorials/lime/test/test_docker_all.py
    tutorials/shap/01.API.shap.py
    tutorials/shap/test/test_docker_all.py
    ```

- Remove the paired one
    ```
    class_scripts/common_utils.py
    class_scripts/for_loop_lessons.py
    class_scripts/gen_slides.py
    class_scripts/gen_slides_test_utils.py
    class_scripts/generate_lecture_commentary.py
    class_scripts/generate_slide_script.py
    class_scripts/test/test_for_loop_lessons.py
    class_scripts/test/test_for_loop_slides.py
    msml610/test/test_gen_slides.py
    ```
- The interesting directories are:
    ```
    class_scripts
    msml610
    data605
    ```
  > pytest class_scripts --cov=class_scripts

Name                                               Stmts   Miss Branch BrPart  Cover
------------------------------------------------------------------------------------
class_scripts/__init__.py                              0      0      0      0   100%
class_scripts/common_utils.py                         59     42      4      0    27%
class_scripts/count_lecture_commentary_pages.py       21     21      4      0     0%
class_scripts/count_lecture_pages.py                  21     21      4      0     0%
class_scripts/count_lecture_slides.py                 71     12     10      1    84%
class_scripts/count_words.py                          29     29      6      0     0%
class_scripts/create_book_toc_from_slides.py         177    177     62      0     0%
class_scripts/extract_png_from_pdf.py                 40     40      4      0     0%
class_scripts/for_loop_lessons.py                    220     83     60     10    60%
class_scripts/for_loop_slides.py                      73     35     10      1    57%
class_scripts/gen_lecture_commentary.py               51     51      2      0     0%
class_scripts/gen_lecture_video_script.py                   57     57      4      0     0%
class_scripts/gen_quizzes.py                          62     62      8      0     0%
class_scripts/gen_slides.py                           67     28     16      1    51%
class_scripts/gen_slides_test_utils.py               259    172     32      1    33%
class_scripts/generate_lecture_commentary.py               153    153     24      0     0%
class_scripts/generate_class_images.py                92     92     18      0     0%
class_scripts/generate_slide_script.py                58     58      8      0     0%
class_scripts/get_lecture_file.py                     20     20      2      0     0%
class_scripts/process_slides.py                      101    101     14      0     0%
class_scripts/slide_check.py                          29     29      4      0     0%
class_scripts/slide_improve.py                        29     29      4      0     0%
class_scripts/slide_reduce.py                         28     28      4      0     0%
class_scripts/slides_utils.py                         54     54     10      0     0%
class_scripts/test/__init__.py                         0      0      0      0   100%
class_scripts/test/test_count_lecture_slides.py       98      0      0      0   100%
class_scripts/test/test_for_loop_lessons.py          281      0      2      0   100%
class_scripts/test/test_for_loop_slides.py            71      0      0      0   100%
class_scripts/test/test_gen_slides.py                103      0      2      0   100%
class_scripts/test/test_gen_slides_validation.py      33      0      6      0   100%
class_scripts/test/test_lesson_round_trip.py          29      0      2      0   100%
------------------------------------------------------------------------------------
TOTAL                                               2386   1394    326     14    39%
