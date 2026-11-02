import datetime
import logging
import os
import sys
from typing import Any, Dict, List
from unittest import mock

import helpers.hcache_simple as hcacsimp
import helpers.hio as hio
import helpers.hllm_cost as hllmcost
import helpers.hunit_test as hunitest
import pytest

import research.Estimate_academic_paper_quality.epq_bib_sources as epqbibso
import research.Estimate_academic_paper_quality.epq_pipeline as epqpipel
import research.Estimate_academic_paper_quality.epq_rater as epqrater
import research.Estimate_academic_paper_quality.epq_report as epqrepor
import research.Estimate_academic_paper_quality.epq_types as epqtypes
import research.Estimate_academic_paper_quality.estimate_paper_quality as repeqpaq
import research.Estimate_academic_paper_quality.test.epq_test_utils as epqteuti

_LOG = logging.getLogger(__name__)

_JSON = "application/json"

_LAZY_S2 = {
    "paperId": "abc123",
    "title": "Lazy Prices",
    "year": 2020,
    "authors": [{"name": "Lauren Cohen"}, {"name": "Christopher Malloy"}],
    "externalIds": {"DOI": "10.1111/jofi.12885", "ArXiv": "2001.00001"},
    "publicationTypes": ["JournalArticle"],
}

_LAZY_ARXIV = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2001.00001v1</id>
<published>2020-01-01T00:00:00Z</published>
<title>Lazy Prices</title>
<author><name>Lauren Cohen</name></author>
</entry></feed>"""


def _json(body: object) -> List[Any]:
    """
    Build a queue with one JSON answer for the local server.
    """
    import json

    return [(200, json.dumps(body).encode("utf-8"), _JSON)]


def _make_payload() -> Dict[str, Any]:
    """
    Build the answer of the fake LLM for the paper `Lazy Prices`.
    """
    unknown: Dict[str, Any] = {"value": "unknown", "evidence": None}
    evidence = {"quote": "Sharpe ratio of 1.4", "section": "Table 4"}
    return {
        "performance": {"value": "Sharpe 1.4", "evidence": evidence},
        "asset_class": {
            "value": "US equities",
            "evidence": {"quote": "CRSP stocks", "section": "Sec. 2"},
        },
        "theoretical_domain": unknown,
        "mechanism": {
            "value": "associational",
            "evidence": {"quote": "made up quote", "section": "Sec. 5"},
        },
        "sharpe_ratios": [
            {"label": "out of sample", "value": "1.4", "evidence": evidence}
        ],
        "annual_returns": [],
        "volatilities": [],
        "max_drawdowns": [],
        "sample_periods": [],
        "universe": unknown,
        "data_assets": unknown,
        "data_frequency": unknown,
        "data_fields": unknown,
        "affiliations": [
            {
                "author": "Lauren Cohen",
                "institution": "Harvard University",
                "kind": "university",
                "evidence": {"quote": "Harvard University", "section": "byline"},
            }
        ],
    }


class _PipelineTestCase(hunitest.TestCase):
    """
    Base of the tests of the pipeline.

    `download_utils.detect_doi()` caches its answers on disk: the cache is off,
    so that the tests write nothing outside the scratch space.
    """

    @pytest.fixture(autouse=True)
    def setup_teardown_test(self):
        """
        Disable the cache of the helpers during each test.
        """
        hcacsimp.enable_caching(False)
        yield
        hcacsimp.enable_caching(True)

    def make_config(
        self, lines: List[str], *, force_extract: bool = False
    ) -> epqpipel.PipelineConfig:
        """
        Write an input file and build the config of a run.

        :param lines: lines of the input file
        :param force_extract: if True, extract again
        :return: config with the output dir in the scratch space
        """
        scratch = self.get_scratch_space()
        input_file = os.path.join(scratch, "list.md")
        hio.to_file(input_file, "\n".join(lines) + "\n")
        config = epqpipel.PipelineConfig(
            input_file=input_file,
            output_dir=os.path.join(scratch, "out"),
            email="",
            model="test-model",
            force_extract=force_extract,
            max_chars=100000,
            today=datetime.date(2026, 10, 28),
            cache_mode="",
            s2_api_key="",
        )
        epqpipel.make_dirs(config)
        return config

    def make_tiers(self) -> epqtypes.AffiliationTiers:
        """
        Build an empty tier list.
        """
        return epqrater.parse_tiers({})


# #############################################################################
# Test_run_stages
# #############################################################################


class Test_run_stages(_PipelineTestCase):
    """
    Test the four stages on a local server, a stub tool, and a fake LLM.
    """

    def test1(self) -> None:
        """
        Test a run from the list to the report, and a rerun that costs nothing.
        """
        # Prepare inputs.
        routes = {
            "/api/query": [
                (200, _LAZY_ARXIV.encode("utf-8"), "application/atom+xml")
            ],
            "/graph/v1/paper/ARXIV:2001.00001": _json(_LAZY_S2),
            "/graph/v1/paper/abc123": _json({"citationCount": 512}),
            "/graph/v1/paper/search": _json({"data": []}),
            "/works": _json({"message": {"items": []}}),
        }
        server = epqteuti.LocalServer(routes)
        self.addCleanup(server.stop)
        scratch = self.get_scratch_space()
        config = self.make_config(["2001.00001", "Unknown Title"])
        sources = epqbibso.BibSources(
            email="",
            s2_api_key="",
            cache_dir=epqpipel.get_cache_dir(config),
            settings=epqteuti.make_settings(server.url),
        )
        tool_path = epqteuti.make_stub_tool(scratch)
        pdf_path = os.path.join(scratch, "source.pdf")
        epqteuti.create_pdf(
            pdf_path,
            ["Harvard University CRSP stocks. Sharpe ratio of 1.4 in table"],
            title_text="Lazy Prices",
        )
        llm_calls: List[Dict[str, Any]] = []
        tracker = hllmcost.LLMCostTracker(model="test-model")

        def _llm_func(**kwargs: Any) -> Any:
            llm_calls.append(kwargs)
            tracker.accumulate_cost(0.02)
            return kwargs["response_format"](**_make_payload())

        env = {
            "EPQ_STUB_CALLS": os.path.join(scratch, "calls.txt"),
            "EPQ_STUB_PDF": pdf_path,
        }
        out = config.output_dir
        # Prepare outputs.
        expected_report = """
        reference,paper_quality,performance,recency,asset_class,theoretical_domain,mechanism,page_length_score,affiliation_score,citations_score,base_name
        "2020, Cohen et al., ""Lazy Prices"" (https://arxiv.org/pdf/2001.00001)",3,Sharpe 1.4,2020,US equities,unknown,associational,1,3,4,2020.Cohen_et_al.Lazy_Prices
        """
        expected_issues = """
        line_no,input,base_name,stage,status,reason,candidates
        2,Unknown Title,,resolve,unresolved,no source has a paper titled 'Unknown Title',
        """
        # Run test: the first run.
        result = epqpipel.run_resolve(config, sources)
        with mock.patch.dict(os.environ, env):
            download_issues = epqpipel.run_download(
                config, list(result.papers), tool_path=tool_path
            )
        extract_issues = epqpipel.run_extract(
            config,
            list(result.papers),
            llm_func=_llm_func,
            cost_tracker=tracker,
            sources=sources,
            tiers=self.make_tiers(),
        )
        epqpipel.run_report(config, list(result.papers))
        # Check outputs.
        self.assertEqual((len(result.papers), len(result.issues)), (1, 1))
        self.assertEqual((download_issues, extract_issues), ([], []))
        self.assert_equal(
            hio.from_file(os.path.join(out, "report.csv")).replace("\r\n", "\n"),
            expected_report,
            dedent=True,
        )
        self.assert_equal(
            hio.from_file(os.path.join(out, "issues.csv")).replace("\r\n", "\n"),
            expected_issues,
            dedent=True,
        )
        self.assert_equal(
            hio.from_file(os.path.join(out, "references.md")),
            '- 2020, Cohen et al., "Lazy Prices" (https://arxiv.org/pdf/2001.00001)\n',
        )
        features = epqrepor.read_features_file(
            epqpipel.get_features_path(config, "2020.Cohen_et_al.Lazy_Prices")
        )
        self.assertAlmostEqual(features.provenance.cost_usd, 0.02)
        self.assertFalse(features.features["mechanism"].evidence.verified)
        self.assertTrue(features.features["performance"].evidence.verified)
        # Run test: a rerun does not call the network, the tool, or the LLM.
        n_requests = len(server.requests)
        n_tool_calls = len(hio.from_file(env["EPQ_STUB_CALLS"]).split("\n"))
        rerun = epqpipel.run_resolve(config, sources)
        with mock.patch.dict(os.environ, env):
            epqpipel.run_download(
                config, list(rerun.papers), tool_path=tool_path
            )
        epqpipel.run_extract(
            config,
            list(rerun.papers),
            llm_func=_llm_func,
            cost_tracker=tracker,
            sources=sources,
            tiers=self.make_tiers(),
        )
        # Check outputs.
        self.assertEqual(len(server.requests), n_requests)
        self.assertEqual(
            len(hio.from_file(env["EPQ_STUB_CALLS"]).split("\n")), n_tool_calls
        )
        self.assertEqual(len(llm_calls), 1)

    def test2(self) -> None:
        """
        Test the error of a lookup for one line does not stop the other lines.
        """

        # Prepare inputs.
        class _Sources(epqteuti.FakeBibSources):
            def lookup_arxiv(self, arxiv_id: str) -> Any:
                if arxiv_id == "1111.11111":
                    raise RuntimeError("Request to 'x' failed: HTTP 500")
                return super().lookup_arxiv(arxiv_id)

        sources = _Sources(
            by_id={
                "arxiv:2001.00001": epqteuti.make_record(
                    source="arxiv", s2_paper_id="", arxiv_id="2001.00001"
                )
            }
        )
        config = self.make_config(["1111.11111", "2001.00001"])
        # Run test.
        result = epqpipel.run_resolve(config, sources)
        # Check outputs.
        self.assertEqual(len(result.papers), 1)
        actual = [(i.line_no, i.status, i.reason) for i in result.issues]
        reason = (
            "no record for arXiv id '1111.11111' "
            "(a source failed: Request to 'x' failed: HTTP 500)"
        )
        expected = [(1, "unresolved", reason)]
        self.assertEqual(actual, expected)

    def test3(self) -> None:
        """
        Test a download that fails is an issue and the run goes on.
        """
        # Prepare inputs.
        config = self.make_config(["x"])
        scratch = self.get_scratch_space()
        papers = [
            epqteuti.make_paper(
                base_name="p1", download_inputs=("https://x.example/fail",)
            ),
            epqteuti.make_paper(base_name="p2", download_inputs=()),
        ]
        tool_path = epqteuti.make_stub_tool(scratch)
        env = {"EPQ_STUB_CALLS": os.path.join(scratch, "calls.txt")}
        # Run test.
        with mock.patch.dict(os.environ, env):
            issues = epqpipel.run_download(config, papers, tool_path=tool_path)
        # Check outputs.
        actual = [(i.base_name, i.status, i.reason) for i in issues]
        expected = [
            (
                "p1",
                "download_failed",
                "https://x.example/fail: Error: HTTP 404 for the input",
            ),
            ("p2", "no_download_link", "no public PDF found"),
        ]
        self.assertEqual(actual, expected)

    def test4(self) -> None:
        """
        Test a paper that fails to extract is an issue and the next one is done.
        """
        # Prepare inputs.
        config = self.make_config(["x"])
        papers_dir = epqpipel.get_papers_dir(config)
        papers = [
            epqteuti.make_paper(base_name="p1"),
            epqteuti.make_paper(base_name="p2"),
        ]
        for name in ["p1", "p2"]:
            epqteuti.create_pdf(
                os.path.join(papers_dir, f"{name}.pdf"), ["Sharpe ratio of 1.4"]
            )
            hio.to_file(os.path.join(papers_dir, f"{name}.md"), "text " * 200)
        tracker = hllmcost.LLMCostTracker(model="test-model")
        n_calls = []

        def _llm_func(**kwargs: Any) -> Any:
            n_calls.append(1)
            if len(n_calls) == 1:
                raise RuntimeError("Rate limit\nlast line of the error")
            return kwargs["response_format"](**_make_payload())

        sources = epqteuti.FakeBibSources()
        # Run test.
        issues = epqpipel.run_extract(
            config,
            papers,
            llm_func=_llm_func,
            cost_tracker=tracker,
            sources=sources,
            tiers=self.make_tiers(),
        )
        # Check outputs.
        actual = [(i.base_name, i.status, i.reason) for i in issues]
        self.assertEqual(
            actual, [("p1", "extraction_failed", "last line of the error")]
        )
        self.assertTrue(os.path.isfile(epqpipel.get_features_path(config, "p2")))

    def test5(self) -> None:
        """
        Test `--force_extract` extracts again a paper with a features file.
        """
        # Prepare inputs.
        config = self.make_config(["x"], force_extract=True)
        papers_dir = epqpipel.get_papers_dir(config)
        epqteuti.create_pdf(
            os.path.join(papers_dir, "p1.pdf"), ["Sharpe ratio of 1.4"]
        )
        hio.to_file(os.path.join(papers_dir, "p1.md"), "text " * 200)
        epqrepor.write_features_file(
            epqpipel.get_features_path(config, "p1"),
            epqteuti.make_paper_features(base_name="p1"),
        )
        tracker = hllmcost.LLMCostTracker(model="test-model")
        n_calls = []

        def _llm_func(**kwargs: Any) -> Any:
            n_calls.append(1)
            return kwargs["response_format"](**_make_payload())

        # Run test.
        epqpipel.run_extract(
            config,
            [epqteuti.make_paper(base_name="p1")],
            llm_func=_llm_func,
            cost_tracker=tracker,
            sources=epqteuti.FakeBibSources(),
            tiers=self.make_tiers(),
        )
        # Check outputs.
        self.assertEqual(len(n_calls), 1)

    def test6(self) -> None:
        """
        Test a features file with an old schema is an issue in the report.
        """
        # Prepare inputs.
        config = self.make_config(["x"])
        paper = epqteuti.make_paper(base_name="p1")
        hio.to_file(
            epqpipel.get_features_path(config, "p1"), '{"schema_version": 99}'
        )
        # Run test.
        epqpipel.run_report(config, [paper])
        # Check outputs.
        issues = hio.from_file(os.path.join(config.output_dir, "issues.csv"))
        self.assertIn("extraction_failed", issues)
        self.assertIn("--force_extract", issues)
        report = hio.from_file(os.path.join(config.output_dir, "report.csv"))
        self.assertEqual(len(report.strip().split("\n")), 1)


# #############################################################################
# Test_estimate_paper_quality
# #############################################################################


class Test_estimate_paper_quality(_PipelineTestCase):
    """
    Test the command line of `estimate_paper_quality.py`.
    """

    def run_main(self, argv: List[str], sources: Any) -> None:
        """
        Run `_main()` with the given flags and a fake client of the APIs.

        :param argv: flags of the script
        :param sources: client of the APIs that replaces the real one
        """
        old_cache_dir = hcacsimp.get_cache_dir()
        self.addCleanup(hcacsimp.set_cache_dir, old_cache_dir)
        parser = repeqpaq._parse()
        with (
            mock.patch.object(sys, "argv", ["estimate_paper_quality.py"] + argv),
            mock.patch.object(epqpipel, "make_sources", return_value=sources),
        ):
            repeqpaq._main(parser)

    def test1(self) -> None:
        """
        Test the stages `resolve` and `report` write the reference list.
        """
        # Prepare inputs.
        config = self.make_config(["2001.00001"])
        sources = epqteuti.FakeBibSources(
            by_id={
                "arxiv:2001.00001": epqteuti.make_record(
                    source="arxiv",
                    s2_paper_id="",
                    arxiv_id="2001.00001",
                    pdf_urls=("https://arxiv.org/pdf/2001.00001",),
                )
            }
        )
        argv = [
            f"--input={config.input_file}",
            f"--output_dir={config.output_dir}",
            "--clear_actions",
            "--action=resolve",
            "--action=report",
        ]
        # Run test.
        self.run_main(argv, sources)
        # Check outputs.
        references = hio.from_file(
            os.path.join(config.output_dir, "references.md")
        )
        self.assert_equal(
            references,
            '- 2020, Cohen et al., "Lazy Prices" (https://arxiv.org/pdf/2001.00001)\n',
        )
        self.assertFalse(
            os.path.exists(
                os.path.join(config.output_dir, "state", "issues.download.json")
            )
        )

    def test2(self) -> None:
        """
        Test the stage `report` alone reads the state of the previous run.
        """
        # Prepare inputs.
        config = self.make_config(["2001.00001"])
        paper = epqteuti.make_paper()
        state_dir = epqpipel.get_state_dir(config)
        epqrepor.write_papers(os.path.join(state_dir, "papers.json"), [paper])
        argv = [
            f"--input={config.input_file}",
            f"--output_dir={config.output_dir}",
            "--clear_actions",
            "--action=report",
        ]
        # Run test.
        self.run_main(argv, epqteuti.FakeBibSources())
        # Check outputs.
        references = hio.from_file(
            os.path.join(config.output_dir, "references.md")
        )
        self.assertIn("Lazy Prices", references)

    def test3(self) -> None:
        """
        Test a missing input file stops the run before any lookup.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        argv = [
            f"--input={scratch}/missing.md",
            f"--output_dir={scratch}/out",
        ]
        # Run test.
        with self.assertRaises(AssertionError):
            self.run_main(argv, epqteuti.FakeBibSources())
