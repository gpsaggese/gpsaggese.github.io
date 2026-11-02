import logging
from typing import Any, Dict, List

import helpers.hprint as hprint
import helpers.hunit_test as hunitest

import research.Estimate_academic_paper_quality.epq_extractor as epqextra

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


# #############################################################################
# Test__drop_references
# #############################################################################


class Test__drop_references(hunitest.TestCase):
    """
    Test `epq_extractor._drop_references()`.
    """

    def helper(self, md_text: str, expected: str) -> None:
        """
        Test helper for `_drop_references()`.

        :param md_text: markdown of the paper
        :param expected: expected markdown without the reference list
        """
        # Run test.
        actual = epqextra._drop_references(md_text)
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
    Test `epq_extractor._prepare_text()`.
    """

    def test1(self) -> None:
        """
        Test a text under the budget is not cut.
        """
        # Run test.
        text, was_cut = epqextra._prepare_text("abc", 10)
        # Check outputs.
        self.assertEqual((text, was_cut), ("abc", False))

    def test2(self) -> None:
        """
        Test a long text keeps 60% from the start and 40% from the end.
        """
        # Prepare inputs.
        md_text = "A" * 100 + "M" * 800 + "Z" * 100
        # Run test.
        text, was_cut = epqextra._prepare_text(md_text, 100)
        # Check outputs.
        self.assertTrue(was_cut)
        self.assertEqual(text, "A" * 60 + "\n[... text cut ...]\n" + "Z" * 40)


# #############################################################################
# Test__locate_quote
# #############################################################################


class Test__locate_quote(hunitest.TestCase):
    """
    Test `epq_extractor._locate_quote()`.
    """

    def helper(self, quote: str, page_texts: List[str], expected: int) -> None:
        """
        Test helper for `_locate_quote()`.

        :param quote: quote of the LLM
        :param page_texts: text of each page
        :param expected: expected page, 0 if not found
        """
        # Run test.
        actual = epqextra._locate_quote(quote, page_texts)
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
            epqextra._locate_quote("", ["text"]),
            epqextra._locate_quote("text", []),
        ]
        # Check outputs.
        self.assertEqual(actual, [0, 0])


# #############################################################################
# Test_extract_features
# #############################################################################


class Test_extract_features(hunitest.TestCase):
    """
    Test `epq_extractor.extract_features()` with a fake LLM function.
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
        return epqextra.extract_features(
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
        result = epqextra.extract_features(
            "# Paper", [], llm_func=_llm_func, max_chars=1000, model="m"
        )
        # Check outputs.
        evidence = result.performance.evidence
        self.assertIsNotNone(evidence)
        self.assertFalse(evidence.verified)
        self.assertEqual(evidence.location, "Table 4")
