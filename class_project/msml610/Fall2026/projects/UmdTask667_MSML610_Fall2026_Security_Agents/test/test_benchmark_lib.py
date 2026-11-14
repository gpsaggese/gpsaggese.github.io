import logging
import os
from typing import Dict, List

import helpers.hio as hio
import helpers.hprint as hprint
import helpers.hunit_test as hunitest

import benchmark_lib as benlib

_LOG = logging.getLogger(__name__)


# #############################################################################
# Snippet builders
# #############################################################################


# Code of a small Flask route; the marked line is the 17th code line.
_VULN_CODE = hprint.dedent(
    """
    import sqlite3

    from flask import Flask, request

    app = Flask(__name__)


    def get_db():
        conn = sqlite3.connect(":memory:")
        return conn


    @app.route("/user")
    def find_user():
        email = request.args.get("email", "")
        conn = get_db()
        query = f"SELECT name FROM users WHERE email = '{email}'"  # {!LINE}
        rows = conn.execute(query).fetchall()
        return {"rows": rows}
    """
)

_SAFE_CODE = hprint.dedent(
    """
    import sqlite3

    from flask import Flask, request

    app = Flask(__name__)


    def get_db():
        conn = sqlite3.connect(":memory:")
        return conn


    @app.route("/user")
    def find_user():
        email = request.args.get("email", "")
        conn = get_db()
        query = "SELECT name FROM users WHERE email = ?"
        rows = conn.execute(query, (email,)).fetchall()
        return {"rows": rows}
    """
)

# 1-based index of the marked line within `_VULN_CODE`.
_VULN_CODE_LINE = 17


def _get_header_fields(snippet_id: str, snippet_type: str) -> Dict[str, str]:
    """
    Return valid header fields for a seeded or secure snippet.
    """
    vulnerable = snippet_type == "seeded"
    fields = {
        "id": snippet_id,
        "type": snippet_type,
        "vulnerable": "true" if vulnerable else "false",
        "cwe": "89",
        "owasp": "A05:2025 Injection",
        "severity": "high" if vulnerable else "none",
        "bug_lines": "[]",
        "attack_scenario": "Attacker sends ' OR '1'='1 as email",
        "remediation": "Use a parameterised query",
        "source": "Written for the unit test",
        "cve_id": "",
        "fix_commit": "",
        "pair_id": "P-001",
        "labelled_by": "Tester",
        "reviewed_by": "",
        "first_pass_agree": "",
    }
    return fields


def _build_snippet(fields: Dict[str, str], code: str) -> str:
    """
    Build the text of a snippet from its header fields and code.

    If the code has a marker and `bug_lines` is `[]`, `bug_lines` is set to the
    marked line in the final file.
    """
    header_lines = ['"""'] + [f"{k}: {v}" for k, v in fields.items()] + ['"""', ""]
    code_lines = code.split("\n")
    if fields.get("bug_lines") == "[]" and benlib.LINE_MARKER in code:
        marked = [
            idx + 1 + len(header_lines)
            for idx, line in enumerate(code_lines)
            if line.rstrip().endswith(benlib.LINE_MARKER)
        ]
        fields = dict(fields, bug_lines=str(marked))
        header_lines = (
            ['"""'] + [f"{k}: {v}" for k, v in fields.items()] + ['"""', ""]
        )
    text = "\n".join(header_lines + code_lines) + "\n"
    return text


def _write_pair(
    benchmark_dir: str,
    *,
    vuln_fields: Dict[str, str],
    safe_fields: Dict[str, str],
    vuln_code: str = _VULN_CODE,
    safe_code: str = _SAFE_CODE,
) -> List[str]:
    """
    Write a seeded snippet and its safe twin into a benchmark dir.

    :return: paths of the two files written
    """
    paths = []
    for fields, code in ((vuln_fields, vuln_code), (safe_fields, safe_code)):
        path = os.path.join(benchmark_dir, fields["type"], f"{fields['id']}.py")
        hio.to_file(path, _build_snippet(fields, code))
        paths.append(path)
    return paths


# #############################################################################
# Test_parse_bug_lines
# #############################################################################


class Test_parse_bug_lines(hunitest.TestCase):
    """
    Test parsing line numbers from the header and CSV formats.
    """

    def test1(self) -> None:
        """
        Test the header format with two lines out of order.
        """
        # Prepare inputs.
        value = "[15, 12]"
        # Prepare outputs.
        expected = [12, 15]
        # Run test.
        actual = benlib.parse_bug_lines(value)
        # Check outputs.
        self.assertEqual(actual, expected)

    def test2(self) -> None:
        """
        Test the CSV format.
        """
        # Prepare inputs.
        value = "12;15"
        # Prepare outputs.
        expected = [12, 15]
        # Run test.
        actual = benlib.parse_bug_lines(value)
        # Check outputs.
        self.assertEqual(actual, expected)

    def test3(self) -> None:
        """
        Test an empty list.
        """
        # Prepare inputs.
        value = "[]"
        # Prepare outputs.
        expected: List[int] = []
        # Run test.
        actual = benlib.parse_bug_lines(value)
        # Check outputs.
        self.assertEqual(actual, expected)


# #############################################################################
# Test_parse_header
# #############################################################################


class Test_parse_header(hunitest.TestCase):
    """
    Test parsing the header docstring of a snippet.
    """

    def test1(self) -> None:
        """
        Test that typed fields are converted and colons in values are kept.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "seeded", "SEED-001.py")
        fields = _get_header_fields("SEED-001", "seeded")
        hio.to_file(path, _build_snippet(fields, _VULN_CODE))
        # Prepare outputs.
        expected_line = 19 + _VULN_CODE_LINE
        # Run test.
        header = benlib.parse_header(path)
        # Check outputs.
        self.assertEqual(header["vulnerable"], True)
        self.assertEqual(header["cwe"], 89)
        self.assertEqual(header["owasp"], "A05:2025 Injection")
        self.assertEqual(header["bug_lines"], [expected_line])


# #############################################################################
# Test_strip_answers
# #############################################################################


class Test_strip_answers(hunitest.TestCase):
    """
    Test removing the header and markers from a snippet.
    """

    def test1(self) -> None:
        """
        Test that the header and marker are removed and the offset is right.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "seeded", "SEED-001.py")
        fields = _get_header_fields("SEED-001", "seeded")
        hio.to_file(path, _build_snippet(fields, _VULN_CODE))
        bug_line = benlib.parse_header(path)["bug_lines"][0]
        # Run test.
        code, offset = benlib.strip_answers(path)
        # Check outputs.
        self.assertNotIn('"""', code)
        self.assertNotIn(benlib.LINE_MARKER, code)
        self.assertTrue(code.startswith("import sqlite3\n"))
        # The stripped line maps back to the marked line.
        stripped_line = bug_line - offset
        actual = code.split("\n")[stripped_line - 1]
        expected = (
            "    query = f\"SELECT name FROM users WHERE email = '{email}'\""
        )
        self.assert_equal(actual, expected)


# #############################################################################
# Test_validate_benchmark
# #############################################################################


class Test_validate_benchmark(hunitest.TestCase):
    """
    Test the benchmark validator, one rule per test.
    """

    def helper(
        self,
        expected: str,
        *,
        vuln_overrides: Dict[str, str],
        safe_overrides: Dict[str, str],
        vuln_code: str = _VULN_CODE,
        safe_code: str = _SAFE_CODE,
    ) -> None:
        """
        Write a pair of snippets and check the validator errors.

        :param expected: expected errors, one per line
        """
        # Prepare inputs.
        benchmark_dir = self.get_scratch_space()
        vuln_fields = dict(_get_header_fields("SEED-001", "seeded"), **vuln_overrides)
        safe_fields = dict(_get_header_fields("SAFE-001", "secure"), **safe_overrides)
        _write_pair(
            benchmark_dir,
            vuln_fields=vuln_fields,
            safe_fields=safe_fields,
            vuln_code=vuln_code,
            safe_code=safe_code,
        )
        # Run test.
        errors = benlib.validate_benchmark(benchmark_dir)
        # Check outputs.
        actual = "\n".join(errors)
        self.assert_equal(actual, expected, dedent=True)

    def test1(self) -> None:
        """
        Test that a valid pair has no errors.
        """
        self.helper("", vuln_overrides={}, safe_overrides={})

    def test2(self) -> None:
        """
        Test that `bug_lines` must match the marked lines.
        """
        expected = """
        seeded/SEED-001.py: bug_lines [10] != marked lines [36]
        """
        self.helper(expected, vuln_overrides={"bug_lines": "[10]"}, safe_overrides={})

    def test3(self) -> None:
        """
        Test that a safe snippet must not carry a marker.
        """
        safe_code = _SAFE_CODE.replace(
            "(email,)).fetchall()", "(email,)).fetchall()  # {!LINE}"
        )
        expected = """
        secure/SAFE-001.py: safe snippet must have empty bug_lines
        secure/SAFE-001.py: safe snippet has markers on lines [37]
        """
        self.helper(
            expected, vuln_overrides={}, safe_overrides={}, safe_code=safe_code
        )

    def test4(self) -> None:
        """
        Test that a comment leaking the label is reported.
        """
        safe_code = _SAFE_CODE.replace(
            '"SELECT name FROM users WHERE email = ?"',
            '"SELECT name FROM users WHERE email = ?"  # safe query',
        )
        expected = """
        secure/SAFE-001.py: label hint in comment on code line 17: 'safe'
        """
        self.helper(
            expected, vuln_overrides={}, safe_overrides={}, safe_code=safe_code
        )

    def test5(self) -> None:
        """
        Test that a snippet with too few code lines is reported.
        """
        safe_code = "\n".join(_SAFE_CODE.split("\n")[-6:])
        expected = """
        secure/SAFE-001.py: 6 code lines, expected 15-80
        """
        self.helper(
            expected, vuln_overrides={}, safe_overrides={}, safe_code=safe_code
        )

    def test6(self) -> None:
        """
        Test that a pair must have one vulnerable and one safe snippet.
        """
        expected = """
        pair 'P-001' must have one vulnerable and one safe snippet, got ['SAFE-001']
        pair 'P-002' must have one vulnerable and one safe snippet, got ['SEED-001']
        """
        self.helper(
            expected, vuln_overrides={"pair_id": "P-002"}, safe_overrides={}
        )

    def test7(self) -> None:
        """
        Test that an out-of-sync `labels.csv` is reported.
        """
        # Prepare inputs.
        benchmark_dir = self.get_scratch_space()
        _write_pair(
            benchmark_dir,
            vuln_fields=_get_header_fields("SEED-001", "seeded"),
            safe_fields=_get_header_fields("SAFE-001", "secure"),
        )
        labels_file = os.path.join(benchmark_dir, "labels.csv")
        df = benlib.write_labels(benchmark_dir, labels_file)
        # Change a label in the CSV only.
        df.loc[0, "severity"] = "low"
        df.to_csv(labels_file, index=False)
        # Run test.
        errors = benlib.validate_benchmark(benchmark_dir, labels_file=labels_file)
        # Check outputs.
        self.assertEqual(len(errors), 1)
        self.assertIn("out of sync", errors[0])


# #############################################################################
# Test_anonymize_benchmark
# #############################################################################


class Test_anonymize_benchmark(hunitest.TestCase):
    """
    Test writing stripped snippets with neutral names.
    """

    def test1(self) -> None:
        """
        Test that file names and contents do not reveal the labels.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        benchmark_dir = os.path.join(scratch_dir, "benchmark")
        dst_dir = os.path.join(scratch_dir, "cases")
        _write_pair(
            benchmark_dir,
            vuln_fields=_get_header_fields("SEED-001", "seeded"),
            safe_fields=_get_header_fields("SAFE-001", "secure"),
        )
        # Run test.
        df = benlib.anonymize_benchmark(benchmark_dir, dst_dir)
        # Check outputs.
        self.assertEqual(sorted(df["snippet_id"]), ["SAFE-001", "SEED-001"])
        for case_file in df["case_file"]:
            self.assertRegex(case_file, r"^case_[0-9a-f]{8}\.py$")
            code = hio.from_file(os.path.join(dst_dir, case_file))
            self.assertNotIn("SEED", code)
            self.assertNotIn("SAFE", code)
            self.assertNotIn(benlib.LINE_MARKER, code)
        self.assertEqual(df["line_offset"].tolist(), [19, 19])
