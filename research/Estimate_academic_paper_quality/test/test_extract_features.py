import logging
import os
import sys
from typing import Any, Callable, Dict, List, Tuple
from unittest import mock

import helpers.hio as hio
import helpers.hllm_cost as hllmcost
import helpers.hprint as hprint
import helpers.hunit_test as hunitest

import research.Estimate_academic_paper_quality.epq_paths as epqpaths
import research.Estimate_academic_paper_quality.epq_report as epqrepor
import research.Estimate_academic_paper_quality.extract_features as extfeat
import research.Estimate_academic_paper_quality.test.epq_test_utils as epqteuti

_LOG = logging.getLogger(__name__)


def _feature(value: str, quote: str = "", section: str = "") -> Dict[str, Any]:
    """
    Build a feature in the shape of the answer of the LLM.
    """
    evidence = {"quote": quote, "section": section} if quote else None
    return {"value": value, "evidence": evidence}


def _fact(label: str, value: str, quote: str, section: str) -> Dict[str, Any]:
    """
    Build a labeled fact in the shape of the answer of the LLM.
    """
    return {
        "label": label,
        "value": value,
        "evidence": {"quote": quote, "section": section},
    }


def _make_llm(
    *, fail_first: bool = False
) -> Tuple[Callable[..., Any], hllmcost.LLMCostTracker, List[int]]:
    """
    Build a fake LLM function that costs 0.02 USD for each call.

    :param fail_first: if True, the first call raises
    :return: function, its cost tracker, and a list with one item per call
    """
    tracker = hllmcost.LLMCostTracker(model="test-model")
    calls: List[int] = []

    def _llm_func(**kwargs: Any) -> Any:
        calls.append(1)
        if fail_first and len(calls) == 1:
            raise RuntimeError("Rate limit\nlast line of the error")
        tracker.accumulate_cost(0.02)
        return kwargs["response_format"](**epqteuti.make_llm_payload())

    return _llm_func, tracker, calls


def _make_config(
    output_dir: str, *, force_extract: bool = False, dry_run: bool = False
) -> extfeat._ExtractConfig:
    """
    Build the config of an extraction run.

    :param output_dir: dir of all the outputs
    :param force_extract: if True, extract again
    :param dry_run: if True, only log what the run would do
    :return: config
    """
    config = extfeat._ExtractConfig(
        output_dir=output_dir,
        model="test-model",
        force_extract=force_extract,
        max_chars=100000,
        dry_run=dry_run,
    )
    return config


def _write_converted_papers(output_dir: str, base_names: List[str]) -> None:
    """
    Write the PDF and the markdown of papers, as `download_papers.py` does.

    :param output_dir: dir of all the outputs
    :param base_names: base names of the papers
    """
    hio.create_dir(epqpaths.get_papers_dir(output_dir), incremental=True)
    for name in base_names:
        epqteuti.create_pdf(
            epqpaths.get_pdf_path(output_dir, name),
            ["Harvard University CRSP stocks. Sharpe ratio of 1.4 in table"],
        )
        hio.to_file(epqpaths.get_markdown_path(output_dir, name), "text " * 200)


# #############################################################################
# Test__drop_references
# #############################################################################


class Test__drop_references(hunitest.TestCase):
    """
    Test `extract_features._drop_references()`.
    """

    def helper(self, md_text: str, expected: str) -> None:
        """
        Test helper for `_drop_references()`.

        :param md_text: markdown of the paper
        :param expected: expected markdown without the reference list
        """
        # Run test.
        actual = extfeat._drop_references(md_text)
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)

    def test1(self) -> None:
        """
        Test the list is dropped up to the next heading of the same level.
        """
        # Prepare inputs.
        md_text = """
        # Paper
        ## 1. Intro
        Text.
        ## 2. References
        [1] Smith.
        ### Sub
        [2] Doe.
        ## Appendix
        Table A.
        """
        # Prepare outputs.
        expected = """
        # Paper
        ## 1. Intro
        Text.
        ## Appendix
        Table A.
        """
        # Run test.
        self.helper(hprint.dedent(md_text), expected)

    def test2(self) -> None:
        """
        Test a list at the end with a bold title is dropped to the end.
        """
        # Prepare inputs.
        md_text = """
        # Paper
        Text.
        # **Bibliography**
        [1] Smith.
        """
        # Prepare outputs.
        expected = """
        # Paper
        Text.
        """
        # Run test.
        self.helper(hprint.dedent(md_text), expected)

    def test3(self) -> None:
        """
        Test a paper with no reference heading is unchanged.
        """
        # Prepare inputs.
        md_text = "# Paper\nText about references in the body."
        # Run test.
        self.helper(md_text, md_text)


# #############################################################################
# Test__prepare_text
# #############################################################################


class Test__prepare_text(hunitest.TestCase):
    """
    Test `extract_features._prepare_text()`.
    """

    def test1(self) -> None:
        """
        Test a text under the budget is not cut.
        """
        # Run test.
        text, was_cut = extfeat._prepare_text("abc", 10)
        # Check outputs.
        self.assertEqual((text, was_cut), ("abc", False))

    def test2(self) -> None:
        """
        Test a long text keeps 60% from the start and 40% from the end.
        """
        # Prepare inputs.
        md_text = "A" * 100 + "M" * 800 + "Z" * 100
        # Run test.
        text, was_cut = extfeat._prepare_text(md_text, 100)
        # Check outputs.
        self.assertTrue(was_cut)
        self.assertEqual(text, "A" * 60 + "\n[... text cut ...]\n" + "Z" * 40)


# #############################################################################
# Test__locate_quote
# #############################################################################


class Test__locate_quote(hunitest.TestCase):
    """
    Test `extract_features._locate_quote()`.
    """

    def helper(self, quote: str, page_texts: List[str], expected: int) -> None:
        """
        Test helper for `_locate_quote()`.

        :param quote: quote of the LLM
        :param page_texts: text of each page
        :param expected: expected page, 0 if not found
        """
        # Run test.
        actual = extfeat._locate_quote(quote, page_texts)
        # Check outputs.
        self.assertEqual(actual, expected)

    def test1(self) -> None:
        """
        Test the first page that contains the quote.
        """
        self.helper(
            "Sharpe ratio of 1.4",
            ["intro", "A  Sharpe\nratio of 1.4 here", "x"],
            2,
        )

    def test2(self) -> None:
        """
        Test a word cut by a hyphen at the end of a line.
        """
        self.helper(
            "out of sample performance", ["out of sam-\nple performance"], 1
        )

    def test3(self) -> None:
        """
        Test a ligature glyph matches its letters.
        """
        self.helper("significant effect", ["a signiﬁcant effect"], 1)

    def test4(self) -> None:
        """
        Test the marks of markdown tables and emphasis are ignored.
        """
        self.helper("| **Sharpe** | 1.4 |", ["Sharpe 1.4"], 1)

    def test5(self) -> None:
        """
        Test a quote that is not in the PDF is not found.
        """
        self.helper("made up by the LLM", ["real text"], 0)

    def test6(self) -> None:
        """
        Test an empty quote and a PDF with no text are not found.
        """
        # Run test.
        actual = [
            extfeat._locate_quote("", ["text"]),
            extfeat._locate_quote("text", []),
        ]
        # Check outputs.
        self.assertEqual(actual, [0, 0])


# #############################################################################
# Test__extract_features
# #############################################################################


class Test__extract_features(hunitest.TestCase):
    """
    Test `extract_features._extract_features()` with a fake LLM function.
    """

    def make_payload(self) -> Dict[str, Any]:
        """
        Build an answer of the LLM for a paper with a few values.

        :return: answer in the shape of the structured output
        """
        unknown = _feature("unknown")
        return {
            "performance": _feature(
                "Sharpe 1.4", "Sharpe ratio of 1.4", "Table 4"
            ),
            "asset_class": _feature("US equities", "US stocks", "Sec. 2"),
            "theoretical_domain": unknown,
            "mechanism": _feature("associational", "made up quote", "Sec. 5"),
            "sharpe_ratios": [
                _fact("out of sample", "1.4", "Sharpe ratio of 1.4", "Table 4"),
                _fact("net of costs", "1.1", "net Sharpe of 1.1", "Table 5"),
            ],
            "annual_returns": [],
            "volatilities": [],
            "max_drawdowns": [],
            "sample_periods": [],
            "universe": _feature("CRSP stocks", "CRSP stocks", "Sec. 2"),
            "data_assets": unknown,
            "data_frequency": _feature("  Unknown "),
            "data_fields": unknown,
            "affiliations": [
                {
                    "author": "Lauren Cohen",
                    "institution": "Harvard University",
                    "kind": "university",
                    "evidence": {
                        "quote": "Harvard University",
                        "section": "byline",
                    },
                }
            ],
        }

    def run_extract(self, max_chars: int, calls: List[Dict[str, Any]]) -> Any:
        """
        Run `extract_features()` on a small paper.

        :param max_chars: budget of characters
        :param calls: list that gets the arguments of each call of the LLM
        :return: extraction result
        """
        payload = self.make_payload()

        def _llm_func(**kwargs: Any) -> Any:
            calls.append(kwargs)
            return kwargs["response_format"](**payload)

        md_text = "# Paper\nCRSP stocks.\n## References\n[1] Smith."
        page_texts = [
            "Harvard University\nUS stocks",
            "CRSP stocks. Sharpe ratio of 1.4 and net Sharpe of 1.1",
        ]
        return extfeat._extract_features(
            md_text,
            page_texts,
            llm_func=_llm_func,
            max_chars=max_chars,
            model="test-model",
        )

    def test1(self) -> None:
        """
        Test the values, the pages, and the unknown values.
        """
        # Prepare inputs.
        calls: List[Dict[str, Any]] = []
        # Prepare outputs.
        expected = """
        performance=Sharpe 1.4|Table 4, p. 2|True
        asset_class=US equities|Sec. 2, p. 1|True
        theoretical_domain=unknown|None
        mechanism=associational|Sec. 5|False
        data_frequency=unknown|None
        universe=CRSP stocks|Sec. 2, p. 2|True
        sharpe=['out of sample|Table 4, p. 2', 'net of costs|Table 5, p. 2']
        affiliation=Harvard University|university|byline, p. 1|True
        """
        # Run test.
        result = self.run_extract(1000, calls)

        # Check outputs.
        def _show(feature: Any) -> str:
            ev = feature.evidence
            tail = "None" if ev is None else f"{ev.location}|{ev.verified}"
            return f"{feature.value}|{tail}"

        sharpe = [
            f"{f.label}|{f.evidence.location}"
            for f in result.facts.sharpe_ratios
        ]
        aff = result.affiliations[0]
        lines = [
            f"performance={_show(result.performance)}",
            f"asset_class={_show(result.asset_class)}",
            f"theoretical_domain={_show(result.theoretical_domain)}",
            f"mechanism={_show(result.mechanism)}",
            f"data_frequency={_show(result.facts.data_frequency)}",
            f"universe={_show(result.facts.universe)}",
            f"sharpe={sharpe}",
            f"affiliation={aff.institution}|{aff.kind}|{aff.evidence.location}|{aff.evidence.verified}",
        ]
        self.assert_equal("\n".join(lines), expected, dedent=True)

    def test2(self) -> None:
        """
        Test the prompt, the text sent to the LLM, and the provenance.
        """
        # Prepare inputs.
        calls: List[Dict[str, Any]] = []
        # Run test.
        result = self.run_extract(1000, calls)
        # Check outputs.
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["user_prompt"], "# Paper\nCRSP stocks.")
        self.assertIn("Copy each quote verbatim", calls[0]["system_prompt"])
        provenance = result.provenance
        self.assertEqual(provenance.model, "test-model")
        self.assertEqual(provenance.chars_sent, len("# Paper\nCRSP stocks."))
        self.assertFalse(provenance.text_was_cut)
        self.assertEqual(len(provenance.prompt_sha1), 40)

    def test3(self) -> None:
        """
        Test the provenance says when the text was cut.
        """
        # Prepare inputs.
        calls: List[Dict[str, Any]] = []
        # Run test.
        result = self.run_extract(10, calls)
        # Check outputs.
        self.assertTrue(result.provenance.text_was_cut)
        self.assertIn("[... text cut ...]", calls[0]["user_prompt"])

    def test4(self) -> None:
        """
        Test with no PDF text every quote is unverified.
        """
        # Prepare inputs.
        payload = self.make_payload()

        def _llm_func(**kwargs: Any) -> Any:
            return kwargs["response_format"](**payload)

        # Run test.
        result = extfeat._extract_features(
            "# Paper", [], llm_func=_llm_func, max_chars=1000, model="m"
        )
        # Check outputs.
        evidence = result.performance.evidence
        self.assertIsNotNone(evidence)
        self.assertFalse(evidence.verified)
        self.assertEqual(evidence.location, "Table 4")


# #############################################################################
# Test__extract
# #############################################################################


class Test__extract(hunitest.TestCase):
    """
    Test the stage of `extract_features.py` with a fake LLM.
    """

    def test1(self) -> None:
        """
        Test the features file has the facts, the affiliations, and the pages.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        _write_converted_papers(output_dir, ["p1"])
        llm_func, tracker, calls = _make_llm()
        # Run test.
        issues = extfeat._extract(
            [epqteuti.make_paper(base_name="p1")],
            _make_config(output_dir),
            llm_func=llm_func,
            cost_tracker=tracker,
        )
        # Check outputs.
        self.assertEqual(issues, [])
        self.assertEqual(len(calls), 1)
        features = epqrepor.read_features_file(
            epqpaths.get_features_path(output_dir, "p1")
        )
        self.assertEqual(features.n_pages, 1)
        actual = [a.institution for a in features.affiliations]
        self.assertEqual(actual, ["Harvard University"])
        self.assertAlmostEqual(features.provenance.cost_usd, 0.02)
        self.assertTrue(features.features["performance"].evidence.verified)
        self.assertFalse(features.features["mechanism"].evidence.verified)
        # The stage does not rate the paper.
        self.assertFalse(
            os.path.exists(epqpaths.get_rating_path(output_dir, "p1"))
        )

    def test2(self) -> None:
        """
        Test a paper that fails to extract is an issue and the next one is done.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        _write_converted_papers(output_dir, ["p1", "p2"])
        llm_func, tracker, _ = _make_llm(fail_first=True)
        papers = [
            epqteuti.make_paper(base_name="p1"),
            epqteuti.make_paper(base_name="p2"),
        ]
        # Run test.
        issues = extfeat._extract(
            papers,
            _make_config(output_dir),
            llm_func=llm_func,
            cost_tracker=tracker,
        )
        # Check outputs.
        actual = [(i.base_name, i.status, i.reason) for i in issues]
        self.assertEqual(
            actual, [("p1", "extraction_failed", "last line of the error")]
        )
        self.assertTrue(
            os.path.isfile(epqpaths.get_features_path(output_dir, "p2"))
        )
        # The issues go to a file, for `report_papers.py`.
        path = epqpaths.get_issues_path(output_dir, "extract")
        self.assertEqual(epqrepor.read_issues(path), issues)

    def test3(self) -> None:
        """
        Test a paper with a features file is not extracted again.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        _write_converted_papers(output_dir, ["p1"])
        epqrepor.write_features_file(
            epqpaths.get_features_path(output_dir, "p1"),
            epqteuti.make_paper_features(base_name="p1"),
        )
        llm_func, tracker, calls = _make_llm()
        # Run test.
        extfeat._extract(
            [epqteuti.make_paper(base_name="p1")],
            _make_config(output_dir),
            llm_func=llm_func,
            cost_tracker=tracker,
        )
        # Check outputs.
        self.assertEqual(calls, [])

    def test4(self) -> None:
        """
        Test `--force_extract` extracts again a paper with a features file.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        _write_converted_papers(output_dir, ["p1"])
        epqrepor.write_features_file(
            epqpaths.get_features_path(output_dir, "p1"),
            epqteuti.make_paper_features(base_name="p1"),
        )
        llm_func, tracker, calls = _make_llm()
        # Run test.
        extfeat._extract(
            [epqteuti.make_paper(base_name="p1")],
            _make_config(output_dir, force_extract=True),
            llm_func=llm_func,
            cost_tracker=tracker,
        )
        # Check outputs.
        self.assertEqual(len(calls), 1)

    def test5(self) -> None:
        """
        Test a paper with no markdown is skipped with no issue.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        llm_func, tracker, calls = _make_llm()
        # Run test.
        issues = extfeat._extract(
            [epqteuti.make_paper(base_name="p1")],
            _make_config(output_dir),
            llm_func=llm_func,
            cost_tracker=tracker,
        )
        # Check outputs.
        self.assertEqual((issues, calls), ([], []))

    def test6(self) -> None:
        """
        Test `dry_run` makes no LLM call and writes no file.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        _write_converted_papers(output_dir, ["p1"])
        llm_func, tracker, calls = _make_llm()
        # Run test.
        extfeat._extract(
            [epqteuti.make_paper(base_name="p1")],
            _make_config(output_dir, dry_run=True),
            llm_func=llm_func,
            cost_tracker=tracker,
        )
        # Check outputs.
        self.assertEqual(calls, [])
        self.assertFalse(
            os.path.exists(epqpaths.get_features_path(output_dir, "p1"))
        )
        self.assertFalse(
            os.path.exists(epqpaths.get_issues_path(output_dir, "extract"))
        )


# #############################################################################
# Test_extract_features_py
# #############################################################################


class Test_extract_features_py(hunitest.TestCase):
    """
    Test the command line of `extract_features.py`.
    """

    def run_main(self, argv: List[str], llm: Tuple[Any, Any, Any]) -> None:
        """
        Run `_main()` with the given flags and a fake LLM.

        :param argv: flags of the script
        :param llm: output of `_make_llm()`
        """
        parser = extfeat._parse()
        with (
            mock.patch.object(sys, "argv", ["extract_features.py"] + argv),
            mock.patch.object(extfeat, "_make_llm_func", return_value=llm[:2]),
        ):
            extfeat._main(parser)

    def test1(self) -> None:
        """
        Test a run writes the features file and the issues of the stage.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        papers_file = os.path.join(scratch, "papers.csv")
        paper = epqteuti.make_paper(base_name="p1")
        epqrepor.write_papers_csv(papers_file, [paper])
        output_dir = os.path.join(scratch, "out")
        _write_converted_papers(output_dir, ["p1"])
        llm = _make_llm()
        argv = [f"--input={papers_file}", f"--output_dir={output_dir}"]
        # Run test.
        self.run_main(argv, llm)
        # Check outputs.
        self.assertEqual(len(llm[2]), 1)
        self.assertTrue(
            os.path.isfile(epqpaths.get_features_path(output_dir, "p1"))
        )
        path = epqpaths.get_issues_path(output_dir, "extract")
        self.assertEqual(epqrepor.read_issues(path), [])

    def test2(self) -> None:
        """
        Test a missing papers CSV stops the run.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        argv = [
            f"--input={scratch}/missing.csv",
            f"--output_dir={scratch}/out",
        ]
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            self.run_main(argv, _make_llm())
        # Check outputs.
        self.assertIn("Run resolve_papers.py", str(cm.exception))

    def test3(self) -> None:
        """
        Test a run before the download stops the run.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        papers_file = os.path.join(scratch, "papers.csv")
        epqrepor.write_papers_csv(papers_file, [epqteuti.make_paper()])
        argv = [f"--input={papers_file}", f"--output_dir={scratch}/out"]
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            self.run_main(argv, _make_llm())
        # Check outputs.
        self.assertIn("Run download_papers.py", str(cm.exception))

    def test4(self) -> None:
        """
        Test `--dry_run` makes no LLM call and writes no file.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        papers_file = os.path.join(scratch, "papers.csv")
        epqrepor.write_papers_csv(
            papers_file, [epqteuti.make_paper(base_name="p1")]
        )
        output_dir = os.path.join(scratch, "out")
        _write_converted_papers(output_dir, ["p1"])
        llm = _make_llm()
        argv = [
            f"--input={papers_file}",
            f"--output_dir={output_dir}",
            "--dry_run",
        ]
        # Run test.
        self.run_main(argv, llm)
        # Check outputs.
        self.assertEqual(llm[2], [])
        self.assertFalse(
            os.path.exists(epqpaths.get_features_path(output_dir, "p1"))
        )
