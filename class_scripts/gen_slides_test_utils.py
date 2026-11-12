"""
Shared utilities for slide generation testing.

Import as:

import class_scripts.gen_slides_test_utils as csgsteut
"""

import logging
import os
import shlex
import sys
from typing import List

import pytest

# To handle better progress bars with pytest buffering output.
from tqdm.auto import tqdm

import class_scripts.common_utils as csccouti
import dev_scripts_helpers.documentation.preprocess_notes as dshdprno
import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hprint as hprint
import helpers.hsystem as hsystem
import helpers.hunit_test as hunitest

_LOG = logging.getLogger(__name__)


# #############################################################################
# LessonDiscovery_TestCase
# #############################################################################


class LessonDiscovery_TestCase(hunitest.TestCase):
    """
    Base class for testing lesson discovery in a course.

    This test case ensures that:
    - All lesson files in a course are properly discovered by filename pattern
    - The expected first lesson file is present among discovered files
    - The course contains at least a minimum number of lessons

    Subclasses must set COURSE_DIR and FIRST_LESSON_FILENAME.
    """

    # Subclasses must override these.
    COURSE_DIR: str = ""
    FIRST_LESSON_FILENAME: str = ""

    def test_check_lesson_discovery(self) -> None:
        """
        Check that the first lesson file is discovered in the course.
        """
        _LOG.debug(hprint.to_str("self.COURSE_DIR self.FIRST_LESSON_FILENAME"))
        hdbg.dassert_ne(self.COURSE_DIR, "")
        hdbg.dassert_ne(self.FIRST_LESSON_FILENAME, "")
        lesson_files = csccouti.get_lesson_files(self.COURSE_DIR)
        self.assertGreater(len(lesson_files), 0)
        # Compare only basenames since discovered paths are full file paths.
        basenames = [os.path.basename(f) for f in lesson_files]
        self.assertIn(self.FIRST_LESSON_FILENAME, basenames)

    def test_check_lesson_count(self) -> None:
        """
        Check that the course has at least the minimum expected number of
        lessons.
        """
        _LOG.debug(hprint.to_str("self.COURSE_DIR"))
        hdbg.dassert_ne(self.COURSE_DIR, "")
        min_expected_lessons = 35
        lessons = csccouti.get_lesson_numbers(self.COURSE_DIR)
        self.assertGreaterEqual(
            len(lessons),
            min_expected_lessons,
            f"{self.COURSE_DIR} should have at least {min_expected_lessons} "
            f"lessons, found {len(lessons)}",
        )
        _LOG.info("%s has %d lessons", self.COURSE_DIR, len(lessons))

    def test_check_lesson_format(self) -> None:
        """
        Check that all lesson numbers match the expected format.
        """
        _LOG.debug(hprint.to_str("self.COURSE_DIR"))
        hdbg.dassert_ne(self.COURSE_DIR, "")
        valid_lesson_pattern = r"^\d+(\.[0-9A-Za-z]+)?$"
        lessons = csccouti.get_lesson_numbers(self.COURSE_DIR)
        for lesson in lessons:
            self.assertRegex(
                lesson,
                valid_lesson_pattern,
                f"Invalid lesson format '{lesson}' in {self.COURSE_DIR}",
            )

    def test_get_lesson_files(self) -> None:
        """
        Check that lesson files are found for the course.
        """
        _LOG.debug(hprint.to_str("self.COURSE_DIR"))
        hdbg.dassert_ne(self.COURSE_DIR, "")
        files = csccouti.get_lesson_files(self.COURSE_DIR)
        self.assertGreater(len(files), 0)
        _LOG.debug("Found %d lesson files", len(files))

    def test_get_lesson_numbers(self) -> None:
        """
        Check that lesson numbers are found for the course.
        """
        _LOG.debug(hprint.to_str("self.COURSE_DIR"))
        hdbg.dassert_ne(self.COURSE_DIR, "")
        numbers = csccouti.get_lesson_numbers(self.COURSE_DIR)
        self.assertGreater(len(numbers), 0)
        _LOG.debug("Found %d lesson numbers", len(numbers))

    def test_collect_all_lessons(self) -> None:
        """
        Check that the course is present in the collected lessons.
        """
        _LOG.debug(hprint.to_str("self.COURSE_DIR"))
        hdbg.dassert_ne(self.COURSE_DIR, "")
        all_lessons = csccouti.collect_all_lessons()
        self.assertIn(self.COURSE_DIR, all_lessons)


# #############################################################################
# Run_preprocess_notes_py_TestCase
# #############################################################################


class Run_preprocess_notes_py_TestCase(hunitest.TestCase):
    """
    Base class for integration tests for `preprocess_notes.py` script.

    This test case verifies that:
    - The script correctly preprocesses lesson files for different courses and
      output types (PDF, HTML, slides)
    - Table of contents options are properly handled during preprocessing
    - Preprocessed output is generated for each specified lesson and written to
      the expected output location
    """

    COURSE_DIR: str = ""

    def _run_preprocess_notes_py(
        self,
        course_dir: str,
        lessons: List[str],
        output_type: str,
        *,
        toc_type: str = "none",
    ) -> None:
        """
        Test preprocessing output for a set of lessons.

        :param course_dir: Course directory (e.g., "data605")
        :param lessons: List of lesson numbers to test
        :param output_type: Either "pdf", "html", or "slides"
        :param toc_type: Type of table of contents ("none", "pandoc_native",
            "navigation", "remove_headers")
        """
        _LOG.debug(hprint.to_str("course_dir lessons output_type toc_type"))
        hdbg.dassert_in(output_type, ("pdf", "html", "slides"))
        scratch_dir = self.get_scratch_space()
        for lesson in tqdm(
            lessons,
            desc=f"Preprocessing {output_type}",
            file=sys.stderr,
            disable=False,
            mininterval=0,
            miniters=1,
        ):
            # Get source file.
            src_name = csccouti.get_source_name(course_dir, lesson)
            input_file = os.path.join(course_dir, "lectures_source", src_name)
            # Use lesson-specific output directory to avoid file conflicts.
            lesson_dir = os.path.join(scratch_dir, f"lesson_{lesson}")
            hio.create_dir(lesson_dir, incremental=True)
            # Prepare output file path.
            output_file = os.path.join(lesson_dir, "preprocessed.md")
            # Prepare command arguments.
            input_arg = shlex.quote(input_file)
            output_arg = shlex.quote(output_file)
            # Build and execute command, e.g.,
            # ```
            # > preprocess_notes.py \
            #     --input='/path/to/lesson.txt' \
            #     --output='/out/notes.md' \
            #     --type=pdf \
            #     --toc_type=none
            # ```
            cmd = (
                f"preprocess_notes.py "
                f"--input={input_arg} "
                f"--output={output_arg} "
                f"--type={output_type} "
                f"--toc_type={toc_type}"
            )
            _LOG.debug("Running command: %s", cmd)
            hsystem.system(cmd)
            # Extract and check output after preprocessing.
            hdbg.dassert_file_exists(output_file)
            content = hio.from_file(output_file)
            self.check_string(content, fuzzy_match=True, tag=lesson)
            _LOG.debug(
                "Verified %s preprocessing for lesson %s",
                output_type,
                lesson,
            )
            #
            sys.stdout.flush()
            sys.stderr.flush()

    # TODO(ai_gp2): Factor out common code
    @pytest.mark.superslow
    def test_preprocess_notes_pdf(self) -> None:
        """
        Test preprocessing notes to PDF format.
        """
        _LOG.debug(hprint.to_str("self.COURSE_DIR"))
        hdbg.dassert_ne(self.COURSE_DIR, "")
        lessons = csccouti.get_lesson_numbers(self.COURSE_DIR)
        # Test PDF output type.
        output_type = "pdf"
        self._run_preprocess_notes_py(self.COURSE_DIR, lessons, output_type)

    @pytest.mark.superslow
    def test_preprocess_notes_html(self) -> None:
        """
        Test preprocessing notes to HTML format.
        """
        _LOG.debug(hprint.to_str("self.COURSE_DIR"))
        hdbg.dassert_ne(self.COURSE_DIR, "")
        lessons = csccouti.get_lesson_numbers(self.COURSE_DIR)
        # Test HTML output type.
        output_type = "html"
        self._run_preprocess_notes_py(self.COURSE_DIR, lessons, output_type)

    @pytest.mark.superslow
    def test_preprocess_notes_slides(self) -> None:
        """
        Test preprocessing notes to slides format.
        """
        _LOG.debug(hprint.to_str("self.COURSE_DIR"))
        hdbg.dassert_ne(self.COURSE_DIR, "")
        lessons = csccouti.get_lesson_numbers(self.COURSE_DIR)
        # Test slides output type.
        output_type = "slides"
        self._run_preprocess_notes_py(self.COURSE_DIR, lessons, output_type)


# #############################################################################
# Run_notes_to_pdf_py_TestCase
# #############################################################################


class Run_notes_to_pdf_py_TestCase(hunitest.TestCase):
    """
    Base class for integration tests for slide generation using `notes_to_pdf.py`.

    # - Tests that Markdown and LaTeX files (PDF or TEX) can be generated from
    #   lesson source notes for a given course
    # - Checks output correctness for multiple lessons
    # - Ensures that preprocessing and file conversion steps are properly
    #   functioning
    # - Verifies both "md" (Markdown→PDF) and "tex" (LaTeX) output types
    # - Runs the conversion script on all selected lessons and asserts that
    #   outputs are produced without error
    """

    COURSE_DIR: str = ""

    def test_slides_engine(self) -> None:
        """
        Check which lessons use the `typst` engine vs the `beamer` (LaTeX)
        engine, as declared via the `// slides_engine=...` metadata directive.
        """
        _LOG.debug(hprint.to_str("self.COURSE_DIR"))
        hdbg.dassert_ne(self.COURSE_DIR, "")
        lessons = csccouti.get_lesson_numbers(self.COURSE_DIR)
        typst_lessons = []
        beamer_lessons = []
        for lesson in tqdm(
            lessons,
            desc="Checking slides engine",
            file=sys.stderr,
            disable=False,
            mininterval=0,
            miniters=1,
        ):
            src_name = csccouti.get_source_name(self.COURSE_DIR, lesson)
            input_file = os.path.join(
                self.COURSE_DIR, "lectures_source", src_name
            )
            lines = hio.from_file(input_file).split("\n")
            metadata, _ = dshdprno.extract_slide_metadata(lines)
            slides_engine = metadata.get("slides_engine", "beamer")
            if slides_engine == "typst":
                typst_lessons.append(lesson)
            else:
                beamer_lessons.append(lesson)
        txt = []
        txt.append(hprint.frame("beamer:"))
        txt.extend(beamer_lessons)
        txt.append(hprint.frame("typst:"))
        txt.extend(typst_lessons)
        actual = "\n".join(txt)
        self.check_string(actual)

    def _run_notes_to_pdf_py(
        self,
        course_dir: str,
        lessons: List[str],
        output_type: str,
    ) -> None:
        """
        Test notes_to_pdf.py output (MD or TeX) for a set of lessons.

        Each lesson declares its rendering engine via a `// slides_engine=typst`
        metadata directive at the top of its source file (default is `beamer`
        when the directive is missing). Lessons whose declared engine doesn't
        match `output_type` (e.g., a `beamer` lesson under a `typ` run) are
        skipped.

        :param course_dir: Course directory (e.g., "data605")
        :param lessons: List of lesson numbers to test
        :param output_type: Either "md" for markdown or "tex" for LaTeX output
        """
        _LOG.debug(hprint.to_str("course_dir lessons output_type"))
        hdbg.dassert_in(output_type, ("tex_pdf", "typ_pdf", "tex", "typ"))
        scratch_dir = self.get_scratch_space()
        for lesson in tqdm(
            lessons,
            desc=f"Testing {output_type} output",
            file=sys.stderr,
            disable=False,
            mininterval=0,
            miniters=1,
        ):
            # Shortcut for debugging.
            if False:
                if lesson < "91":
                    _LOG.warning("Skip lesson '%s'", lesson)
                    continue
            # Get source file.
            src_name = csccouti.get_source_name(course_dir, lesson)
            input_file = os.path.join(course_dir, "lectures_source", src_name)
            # Use `extract_slide_metadata()` to decide if the lesson slides are
            # typst or latex, and skip lessons whose declared engine doesn't
            # match `output_type` under test.
            lines = hio.from_file(input_file).split("\n")
            metadata, _ = dshdprno.extract_slide_metadata(lines)
            slides_engine = metadata.get("slides_engine", "beamer")
            is_typst_output_type = output_type in ("typ_pdf", "typ")
            _LOG.debug(
                hprint.to_str("slides_engine output_type is_typst_output_type")
            )
            if (slides_engine == "typst") != is_typst_output_type:
                _LOG.info(
                    "Skip lesson '%s': slides_engine='%s' doesn't match "
                    "output_type='%s'",
                    lesson,
                    slides_engine,
                    output_type,
                )
                continue
            # Use lesson-specific output directory to avoid file conflicts.
            lesson_dir = os.path.join(scratch_dir, f"lesson_{lesson}")
            hio.create_dir(lesson_dir, incremental=True)
            # Two files are generated and are important:
            # - `output_file` is the final rendered artifact (PDF/TeX/Typst)
            # - `to_check_file` is the intermediate file whose content is checked
            # against the expected test output via `check_string()`.
            if output_type == "tex_pdf":
                output_file = os.path.join(lesson_dir, "tex_output.pdf")
                # No file for `check_string()`.
                to_check_file = ""
                extra_args = ""
            elif output_type == "typ_pdf":
                output_file = os.path.join(lesson_dir, "typ_output.pdf")
                # No file for `check_string()`.
                to_check_file = ""
                extra_args = ""
            elif output_type == "tex":
                output_file = os.path.join(lesson_dir, "output.tex")
                # The file to check_string is the output tex file.
                to_check_file = output_file
                # Only generate the tex file.
                extra_args = "--no_pdf"
            elif output_type == "typ":
                output_file = os.path.join(lesson_dir, "output.typ")
                # The file to check_string is the output typ file.
                to_check_file = output_file
                # Only generate the typ file.
                extra_args = "--no_pdf"
            else:
                raise ValueError(f"Unknown output_type: {output_type}")
            # Prepare command arguments.
            input_arg = shlex.quote(input_file)
            output_arg = shlex.quote(output_file)
            output_type_arg = "slides"
            toc_type_arg = "navigation"
            cleanup_action = "cleanup_after"
            open_action = "open_pdf"
            # Build skip_action arguments.
            all_skip_actions = [cleanup_action, open_action]
            skip_action_args = " ".join(
                f"--skip_action={action}" for action in all_skip_actions
            )
            # Build and execute command.
            cmd = (
                f"notes_to_pdf.py "
                f"--input={input_arg} "
                f"--output={output_arg} "
                f"--type={output_type_arg} "
                f"--toc_type={toc_type_arg} "
                f"--slides_engine={slides_engine} "
                f"{skip_action_args} "
                f"{extra_args}"
            )
            _LOG.debug("Running command: %s", cmd)
            hsystem.system(cmd)
            # Check output.
            _LOG.debug(hprint.to_str("output_file"))
            hdbg.dassert_file_exists(output_file)
            # Process file to check, if any.
            _LOG.debug(hprint.to_str("to_check_file"))
            if to_check_file:
                content = hio.from_file(to_check_file)
                self.check_string(content, fuzzy_match=True, tag=lesson)
            #
            _LOG.debug("Verified %s output for lesson %s", output_type, lesson)
            sys.stdout.flush()

    # TODO(ai_gp2): Factor out common code

    @pytest.mark.superslow
    def test_tex_output(self) -> None:
        """
        Test converting and saving notes to TeX.

        > notes_to_pdf.py \
            --input=msml610/lectures_source/Lesson00-Class.smd
            --output=.../tmp.scratch/lesson_00/output.tex \
            --type=slides \
            --toc_type=navigation \
            --skip_action=cleanup_after \
            --skip_action=open_pdf \
            --no_pdf
        """
        _LOG.debug(hprint.to_str("self.COURSE_DIR"))
        hdbg.dassert_ne(self.COURSE_DIR, "")
        lessons = csccouti.get_lesson_numbers(self.COURSE_DIR)
        output_type = "tex"
        self._run_notes_to_pdf_py(self.COURSE_DIR, lessons, output_type)

    @pytest.mark.superslow
    def test_typ_output(self) -> None:
        """
        Test converting and saving notes to Typst.
        """
        _LOG.debug(hprint.to_str("self.COURSE_DIR"))
        hdbg.dassert_ne(self.COURSE_DIR, "")
        lessons = csccouti.get_lesson_numbers(self.COURSE_DIR)
        output_type = "typ"
        self._run_notes_to_pdf_py(self.COURSE_DIR, lessons, output_type)

    @pytest.mark.superslow
    def test_tex_pdf(self) -> None:
        """
        Test converting latex notes to PDF: do not save the result but only
        check that the PDF is generated.

        > notes_to_pdf.py \
            --input=msml610/lectures_source/Lesson00-Class.smd
            --output=...tmp.scratch/lesson_00/output.pdf
            --type=slides \
            --toc_type=navigation \
            --skip_action=cleanup_after \
            --skip_action=open_pdf
        """
        _LOG.debug(hprint.to_str("self.COURSE_DIR"))
        hdbg.dassert_ne(self.COURSE_DIR, "")
        lessons = csccouti.get_lesson_numbers(self.COURSE_DIR)
        output_type = "tex_pdf"
        self._run_notes_to_pdf_py(
            self.COURSE_DIR,
            lessons,
            output_type,
        )

    @pytest.mark.superslow
    def test_typ_pdf(self) -> None:
        """
        Test converting typist notes to PDF: do not save the result but only
        check that the PDF is generated.

        > notes_to_pdf.py \
            --input=msml610/lectures_source/Lesson00-Class.smd
            --output=...tmp.scratch/lesson_00/output.pdf
            --type=slides \
            --toc_type=navigation \
            --skip_action=cleanup_after \
            --skip_action=open_pdf
        """
        _LOG.debug(hprint.to_str("self.COURSE_DIR"))
        hdbg.dassert_ne(self.COURSE_DIR, "")
        lessons = csccouti.get_lesson_numbers(self.COURSE_DIR)
        output_type = "typ_pdf"
        self._run_notes_to_pdf_py(
            self.COURSE_DIR,
            lessons,
            output_type,
        )


# #############################################################################
# Run_gen_slides_py_TestCase
# #############################################################################


class Run_gen_slides_py_TestCase(hunitest.TestCase):
    """
    Base class for testing `gen_slides.py` script with course-specific lessons.

    Tests performed:
    - Verify that `gen_slides.py` can generate TeX output for the first lesson.
    - Verify that `gen_slides.py` can generate TeX output for the second lesson.
    - Render and check that TeX output is generated for all lessons in the course.
    """

    COURSE_DIR: str = ""
    FIRST_LESSON: str = ""
    SECOND_LESSON: str = ""

    @staticmethod
    def _run_gen_slides(course_dir: str, lesson: str) -> None:
        """
        Run gen_slides for a lesson, generating only TeX output.
        """
        _LOG.debug(hprint.to_str("course_dir lesson"))
        cmd = (
            f"gen_slides.py -i {course_dir}/{lesson} "
            '--notes_to_pdf_args="--skip_action open_pdf --no_pdf"'
        )
        hsystem.system(cmd)

    @pytest.mark.slow
    def test_gen_slides_first_lesson(self) -> None:
        """
        Test generating slides for the first lesson.
        """
        _LOG.debug(hprint.to_str("self.COURSE_DIR self.FIRST_LESSON"))
        hdbg.dassert_ne(self.COURSE_DIR, "")
        self._run_gen_slides(self.COURSE_DIR, self.FIRST_LESSON)

    @pytest.mark.slow
    def test_gen_slides_second_lesson(self) -> None:
        """
        Test generating slides for the second lesson.
        """
        _LOG.debug(hprint.to_str("self.COURSE_DIR self.SECOND_LESSON"))
        hdbg.dassert_ne(self.COURSE_DIR, "")
        hdbg.dassert_ne(self.SECOND_LESSON, "")
        self._run_gen_slides(self.COURSE_DIR, self.SECOND_LESSON)
