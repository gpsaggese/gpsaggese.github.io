"""
Import as:

import test.test_fetch_coding_scores as ttfecosc
"""

import logging
import os
import re
import tarfile
from typing import List

import helpers.hio as hio
import helpers.hunit_test as hunitest

import fetch_coding_scores as fecosc

_LOG = logging.getLogger(__name__)

# The dirs of the phase-1 archive.
_SCORE_DIRS = ("data/coding_runs/", "data/coding_runs_pilot/")


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
# Test_archives
# #############################################################################


class Test_archives(hunitest.TestCase):
    """
    Test the list of archives in `fetch_coding_scores.ARCHIVES`.
    """

    def test1(self) -> None:
        """
        Test that each archive has a full SHA-256 and dirs of its own.
        """
        # Prepare inputs.
        archives = fecosc.ARCHIVES
        # Run test.
        all_dirs = [d for _, _, dirs in archives.values() for d in dirs]
        # Check outputs.
        for file_name, (file_id, sha256, dirs) in archives.items():
            self.assertTrue(file_name.endswith(".tar.gz"), file_name)
            self.assertTrue(file_id, file_name)
            self.assertRegex(sha256, re.compile(r"^[0-9a-f]{64}$"))
            for dir_name in dirs:
                self.assertTrue(dir_name.startswith("data/"), dir_name)
                self.assertTrue(dir_name.endswith("/"), dir_name)
        # No 2 archives write into the same dir.
        self.assertEqual(len(all_dirs), len(set(all_dirs)))


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
        extracted = fecosc.extract_scores(path, dst_dir, _SCORE_DIRS)
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
            fecosc.extract_scores(path, dst_dir, _SCORE_DIRS)
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
            fecosc.extract_scores(path, dst_dir, _SCORE_DIRS)
        self.assertFalse(os.path.exists(dst_dir))

    def test4(self) -> None:
        """
        Test that the phase-2 archive cannot write into the phase-1 dirs.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        path = _make_archive(scratch_dir, ["data/coding_runs/x_t0.csv"])
        dst_dir = os.path.join(scratch_dir, "project")
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            fecosc.extract_scores(path, dst_dir, ("data/coding_runs_phase2/",))
        self.assertFalse(os.path.exists(dst_dir))
