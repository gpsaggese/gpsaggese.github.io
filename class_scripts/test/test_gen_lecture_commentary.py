"""
Unit tests for gen_lecture_commentary.py.

Import as:

import class_scripts.test.test_gen_lecture_commentary as csttgelcom
"""

import logging
from unittest import mock

import pytest

import helpers.hprint as hprint
import helpers.hunit_test as hunitest

# `gen_lecture_commentary.py` is a `uv run` script that requires `pdf2image`.
pytest.importorskip("pdf2image")
import class_scripts.gen_lecture_commentary as csgeleco

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_get_image_extension
# #############################################################################


class Test_get_image_extension(hunitest.TestCase):
    """
    Test `class_scripts.gen_lecture_commentary.get_image_extension()`
    function.
    """

    def helper(self, image_type: str, expected: str) -> None:
        """
        Test helper for `get_image_extension()`.

        :param image_type: image type to test
        :param expected: expected file extension
        """
        # Run test.
        actual = csgeleco.get_image_extension(image_type)
        # Check outputs.
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test happy path: "png" image type maps to the "png" extension.
        """
        # Prepare inputs.
        image_type = "png"
        # Prepare outputs.
        expected = "png"
        # Run test.
        self.helper(image_type, expected)

    def test2(self) -> None:
        """
        Test happy path: "jpg" image type maps to the "jpg" extension.
        """
        # Prepare inputs.
        image_type = "jpg"
        # Prepare outputs.
        expected = "jpg"
        # Run test.
        self.helper(image_type, expected)


# #############################################################################
# Test__generate_slide_commentary
# #############################################################################


class Test__generate_slide_commentary(hunitest.TestCase):
    """
    Test `class_scripts.gen_lecture_commentary._generate_slide_commentary()`
    function.
    """

    def test1(self) -> None:
        """
        Test happy path: the slide content (with no images) is forwarded to
        `helpers.hllm.get_completion()` and its response is returned
        unchanged.
        """
        # Prepare inputs.
        slide_content = """
        * Slide 1
        Some text without images.
        """
        slide_content = hprint.dedent(slide_content)
        system_prompt = "You are a helpful assistant."
        model = "gpt-4"
        llm_backend = "hllm"
        # Prepare outputs.
        expected = "Commentary about slide 1."
        # Run test.
        with mock.patch(
            "helpers.hllm.get_completion", return_value=expected
        ) as mock_get_completion:
            actual = csgeleco._generate_slide_commentary(
                slide_content, system_prompt, model, llm_backend
            )
        # Check outputs.
        self.assert_equal(actual, expected)
        mock_get_completion.assert_called_once_with(
            user_prompt=slide_content,
            system_prompt=system_prompt,
            model=model,
            cache_mode="NORMAL",
            temperature=0.1,
            images_as_base64=(),
        )
