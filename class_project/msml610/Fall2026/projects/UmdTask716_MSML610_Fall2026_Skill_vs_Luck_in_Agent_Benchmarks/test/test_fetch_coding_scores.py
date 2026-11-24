"""
Import as:

import test.test_fetch_coding_scores as ttfecosc
"""

import logging
import os
import tarfile
from typing import List

import helpers.hio as hio
import helpers.hunit_test as hunitest

import fetch_coding_scores as fecosc

_LOG = logging.getLogger(__name__)


def _make_archive(dir_name: str, names: List[str]) -> str:
    """
    Build a `.tar.gz` archive with 1 small text file per name.

    :param dir_name: dir where the archive and its files are written
    :param names: paths of the files inside the archive
    :return: path of the archive
    """
    path = os.path.join(dir_name, "scores.tar.gz")
    with tarfile.open(path, "w:gz") as archive:
        for index, name in enumerate(names):
            # Each file holds its own name, so the content can be checked.
            file_path = os.path.join(dir_name, "src", f"file_{index}.txt")
            hio.to_file(file_path, name)
            archive.add(file_path, arcname=name)
    return path


# #############################################################################
# Test_get_sha256
# #############################################################################


class Test_get_sha256(hunitest.TestCase):
    """
    Test `fetch_coding_scores.get_sha256()`.
    """

    def test1(self) -> None:
        """
        Test the digest of a known text.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "abc.txt")
        hio.to_file(path, "abc")
        # Prepare outputs.
        expected = (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )
        # Run test.
        digest = fecosc.get_sha256(path)
        # Check outputs.
        self.assert_equal(digest, expected)


# #############################################################################
# Test_extract_scores
# #############################################################################


class Test_extract_scores(hunitest.TestCase):
    """
    Test `fetch_coding_scores.extract_scores()`.
    """

    def test1(self) -> None:
        """
        Test that the files of both score dirs are extracted.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        # Plain-text files, since `hio.from_file()` unzips any `.gz` file.
        names = [
            "data/coding_runs/local_humaneval_a_t0.8.csv",
            "data/coding_runs_pilot/local_humaneval_a_t0.csv",
        ]
        path = _make_archive(scratch_dir, names)
        dst_dir = os.path.join(scratch_dir, "project")
        # Run test.
        extracted = fecosc.extract_scores(path, dst_dir)
        # Check outputs.
        self.assertEqual(extracted, names)
        for name in names:
            text = hio.from_file(os.path.join(dst_dir, name))
            self.assertEqual(text, name)

    def test2(self) -> None:
        """
        Test that a file outside the score dirs is refused.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        path = _make_archive(scratch_dir, ["data/other/notes.txt"])
        dst_dir = os.path.join(scratch_dir, "project")
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            fecosc.extract_scores(path, dst_dir)
        self.assertFalse(os.path.exists(dst_dir))

    def test3(self) -> None:
        """
        Test that a path that climbs out of the score dirs is refused.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        path = _make_archive(scratch_dir, ["data/coding_runs/../../evil.txt"])
        dst_dir = os.path.join(scratch_dir, "project")
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            fecosc.extract_scores(path, dst_dir)
        self.assertFalse(os.path.exists(dst_dir))
