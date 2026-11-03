import csv
import datetime
import json
import logging
import os
from typing import Any, List
from unittest import mock

import helpers.hcache_simple as hcacsimp
import helpers.hio as hio
import helpers.hllm_cost as hllmcost
import helpers.hunit_test as hunitest
import pytest

import research.Estimate_academic_paper_quality.download_papers as dowpapers
import research.Estimate_academic_paper_quality.epq_bib_sources as epqbibso
import research.Estimate_academic_paper_quality.epq_paths as epqpaths
import research.Estimate_academic_paper_quality.epq_report as epqrepor
import research.Estimate_academic_paper_quality.extract_features as extfeat
import research.Estimate_academic_paper_quality.report_papers as repopapers
import research.Estimate_academic_paper_quality.resolve_papers as resopapers
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
    return [(200, json.dumps(body).encode("utf-8"), _JSON)]


# #############################################################################
# Test_end_to_end
# #############################################################################


class Test_end_to_end(hunitest.TestCase):
    """
    Test the four executables in order, linked only by files on disk.
    """

    @pytest.fixture(autouse=True)
    def setup_teardown_test(self):
        """
        Disable the cache of the helpers during each test.

        `download_utils.detect_doi()` caches its answers on disk: the cache is
        off, so that the tests write nothing outside the scratch space.
        """
        hcacsimp.enable_caching(False)
        yield
        hcacsimp.enable_caching(True)

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
        output_dir = os.path.join(scratch, "out")
        input_csv = os.path.join(scratch, "papers_in.csv")
        with open(input_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["input"])
            writer.writerows([["2001.00001"], ["Unknown Title"]])
        papers_csv = os.path.join(scratch, "papers.csv")
        issues_csv = epqpaths.get_resolve_issues_path(papers_csv)
        sources = epqbibso.BibSources(
            email="",
            s2_api_key="",
            cache_dir=epqpaths.get_cache_dir(output_dir),
            settings=epqteuti.make_settings(server.url),
        )
        tool_path = epqteuti.make_stub_tool(scratch)
        pdf_path = os.path.join(scratch, "source.pdf")
        epqteuti.create_pdf(
            pdf_path,
            ["Harvard University CRSP stocks. Sharpe ratio of 1.4 in table"],
            title_text="Lazy Prices",
        )
        tracker = hllmcost.LLMCostTracker(model="test-model")
        llm_calls: List[Any] = []

        def _llm_func(**kwargs: Any) -> Any:
            llm_calls.append(kwargs)
            tracker.accumulate_cost(0.02)
            return kwargs["response_format"](**epqteuti.make_llm_payload())

        config = extfeat._ExtractConfig(
            output_dir=output_dir,
            model="test-model",
            force_extract=False,
            max_chars=100000,
            dry_run=False,
        )
        env = {
            "EPQ_STUB_CALLS": os.path.join(scratch, "calls.txt"),
            "EPQ_STUB_PDF": pdf_path,
        }

        def _run_stages() -> Any:
            """
            Run the four executables in order, as `estimate_paper_quality.sh`.
            """
            result = resopapers._resolve(
                input_csv, papers_csv, issues_csv, sources
            )
            # Each executable reads the papers CSV from the disk.
            papers = epqrepor.read_papers_csv(papers_csv)
            with mock.patch.dict(os.environ, env):
                download_issues = dowpapers._download(
                    papers,
                    output_dir,
                    tool_path=tool_path,
                    log_level="INFO",
                    dry_run=False,
                )
            extract_issues = extfeat._extract(
                papers, config, llm_func=_llm_func, cost_tracker=tracker
            )
            rate_issues = repopapers._rate(
                papers,
                output_dir,
                sources=sources,
                tiers=repopapers._parse_tiers({}),
                today=datetime.date(2026, 10, 28),
                refresh_citations=False,
                dry_run=False,
            )
            repopapers._report(papers, output_dir, issues_csv, dry_run=False)
            return result, download_issues, extract_issues, rate_issues

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
        result, download_issues, extract_issues, rate_issues = _run_stages()
        # Check outputs.
        self.assertEqual((len(result.papers), len(result.issues)), (1, 1))
        self.assertEqual(
            (download_issues, extract_issues, rate_issues), ([], [], [])
        )
        self.assert_equal(
            hio.from_file(os.path.join(output_dir, "report.csv")).replace(
                "\r\n", "\n"
            ),
            expected_report,
            dedent=True,
        )
        self.assert_equal(
            hio.from_file(os.path.join(output_dir, "issues.csv")).replace(
                "\r\n", "\n"
            ),
            expected_issues,
            dedent=True,
        )
        self.assert_equal(
            hio.from_file(os.path.join(output_dir, "references.md")),
            '- 2020, Cohen et al., "Lazy Prices" (https://arxiv.org/pdf/2001.00001)\n',
        )
        features = epqrepor.read_features_file(
            epqpaths.get_features_path(
                output_dir, "2020.Cohen_et_al.Lazy_Prices"
            )
        )
        self.assertAlmostEqual(features.provenance.cost_usd, 0.02)
        self.assertFalse(features.features["mechanism"].evidence.verified)
        self.assertTrue(features.features["performance"].evidence.verified)
        # Run test: a rerun does not call the network, the tool, or the LLM.
        n_requests = len(server.requests)
        n_tool_calls = len(hio.from_file(env["EPQ_STUB_CALLS"]).split("\n"))
        _run_stages()
        # Check outputs.
        self.assertEqual(len(server.requests), n_requests)
        self.assertEqual(
            len(hio.from_file(env["EPQ_STUB_CALLS"]).split("\n")), n_tool_calls
        )
        self.assertEqual(len(llm_calls), 1)
