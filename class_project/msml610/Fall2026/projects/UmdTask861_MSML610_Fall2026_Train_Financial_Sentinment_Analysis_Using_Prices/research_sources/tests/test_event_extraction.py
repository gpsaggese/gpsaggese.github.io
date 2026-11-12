"""Synthetic semantic edge cases; no network, broker, or case-review fixtures."""

import importlib.util
from pathlib import Path
import unittest


_path = Path(__file__).resolve().parents[1] / "event_extraction.py"
_spec = importlib.util.spec_from_file_location("event_extraction", _path)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)
extract_article, cluster_events = module.extract_article, module.cluster_events


def article(headline, *, id="a", text="", symbols=None, day="2026-10-01", updated=None):
    return {"id": id, "headline": headline, "text": text, "symbols": symbols or ["NVDA"],
            "created_at": f"{day}T12:00:00Z", "updated_at": updated or f"{day}T12:00:00Z",
            "url": "https://example.test/article", "retrieved_at": f"{day}T12:05:00Z"}


class ExtractionTests(unittest.TestCase):
    def test_tag_only_does_not_receive_other_company_event(self):
        result = extract_article(article("Rocket Lab signs a major launch contract", symbols=["PLTR"]), "PLTR")
        self.assertEqual((result["relevance"], result["event_type"], result["stance"]), ("tag_only", "other", "unknown"))
        self.assertEqual(result["evidence"], [])

    def test_secondary_company_trading_is_not_headline_contract(self):
        source = article("Rocket Lab wins a major launch contract", text="The fund sold Palantir shares while Rocket Lab won a launch contract.", symbols=["PLTR"])
        result = extract_article(source, "PLTR")
        self.assertEqual((result["relevance"], result["event_type"]), ("body", "trading_activity"))
        self.assertNotEqual(result["stance"], "bearish")
        self.assertTrue(all("launch contract" not in span["text"] for span in result["evidence"]))

    def test_cloud_credits_are_not_revenue(self):
        result = extract_article(article("Nvidia and Startup A announce a partnership with $20 million in cloud credits"), "NVDA")
        self.assertEqual(result["event_type"], "partnership")
        self.assertIn("not booked revenue", result["summary"])
        self.assertEqual(result["stance"], "unknown")

    def test_nonrenewal_is_not_a_new_award(self):
        result = extract_article(article("Tesla will not renew its battery supply contract", symbols=["TSLA"]), "TSLA")
        self.assertEqual(result["event_type"], "contract_change")
        self.assertIn("non-renewal", result["summary"])
        self.assertEqual(result["stance"], "unknown")

    def test_analyst_target_is_opinion_not_forecast(self):
        result = extract_article(article("Morgan Stanley raises Nvidia price target to $210 from $190"), "NVDA")
        self.assertEqual((result["event_type"], result["claim_kind"], result["stance"]), ("analyst_revision", "opinion", "bullish"))
        self.assertIn("not return forecasts", result["summary"])
        self.assertIn("Morgan Stanley raised", result["summary"])

    def test_rumor_remains_unconfirmed(self):
        result = extract_article(article("Nvidia reportedly in talks for a new supply agreement"), "NVDA")
        self.assertEqual(result["claim_kind"], "unknown")
        self.assertIn("Unconfirmed", result["summary"])

    def test_product_is_not_an_investment_stance(self):
        result = extract_article(article("Nvidia launches a new chip platform"), "NVDA")
        self.assertEqual((result["event_type"], result["stance"]), ("product", "unknown"))

    def test_negation_is_not_positive_stance_or_confirmed_product(self):
        result = extract_article(article("An analyst is not bullish on Nvidia"), "NVDA")
        self.assertEqual(result["stance"], "unknown")
        denied = extract_article(article("Nvidia has not announced a launch agreement"), "NVDA")
        self.assertEqual(denied["claim_kind"], "unknown")

    def test_evidence_offsets_address_original_text(self):
        source = article("Market update", text="前言。\n" + "Long prefix " * 40 + "Nvidia launches a chip platform with Vendor Z.\nOther news.")
        result = extract_article(source, "NVDA")
        for evidence in result["evidence"]:
            self.assertEqual(source[evidence["field"]][evidence["start"]:evidence["end"]], evidence["text"])
        self.assertTrue(result["evidence"])

    def test_company_abbreviations_do_not_cut_the_event(self):
        source = article("Update", text="Nvidia Corp. announced a product launch in the U.S. market while Tesla Inc. won a supply contract.")
        clauses = module.company_clauses(source, "NVDA")
        self.assertEqual(len(clauses), 1)
        self.assertIn("Corp. announced", clauses[0])
        self.assertIn("U.S. market", clauses[0])
        self.assertNotIn("Tesla", clauses[0])
        self.assertEqual(extract_article(source, "NVDA")["event_type"], "product")

    def test_aws_alias_is_company_specific(self):
        result = extract_article(article("AWS launches new cloud service", symbols=["AMZN"]), "AMZN")
        self.assertEqual((result["relevance"], result["event_type"]), ("headline", "product"))

    def test_conditional_outlook_does_not_make_reported_rating_a_rumor(self):
        result = extract_article(article("Barclays upgraded Nvidia to Buy as demand could grow"), "NVDA")
        self.assertEqual(result["claim_kind"], "opinion")
        self.assertNotIn("Unconfirmed", result["summary"])
        self.assertTrue(any("Conditional outlook" in item for item in result["limitations"]))

    def test_missing_fields_are_safe(self):
        result = extract_article({}, "NVDA")
        self.assertEqual((result["relevance"], result["claim_kind"]), ("none", "unknown"))
        self.assertEqual(result["extraction_method"], "rules_v1")


class ClusteringTests(unittest.TestCase):
    def test_duplicate_analyst_report_merges_and_dates_are_source_times(self):
        title = "Morgan Stanley raises Nvidia price target to $210 from $190 on strong datacenter demand"
        first = extract_article(article(title, id="first"), "NVDA")
        second = extract_article(article(title + ".", id="second", updated="2026-10-01T19:00:00Z"), "NVDA")
        second["published_at"] = "2026-10-01T14:00:00Z"
        clusters = cluster_events([second, first])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["article_ids"], ["first", "second"])
        self.assertEqual(clusters[0]["earliest_reported_at"], first["published_at"])
        self.assertEqual(clusters[0]["latest_updated_at"], second["updated_at"])

    def test_roundup_and_followup_merge_on_broker_action_destination(self):
        first = article("Barclays Upgrades Nvidia to Buy, Announces $210 Price Target", id="a")
        roundup = article("This Nvidia Analyst Turns Bullish", id="b", text="Barclays analyst A. Reader upgraded Nvidia Corp. from Neutral to Buy and announced a $210 price target. Another broker upgraded a different company.")
        followup = article("Nvidia gets Barclays upgrade as demand grows", id="c", text="Barclays upgraded Nvidia Corp. to Buy from Neutral on Thursday.")
        results = [extract_article(a, "NVDA") for a in [first, roundup, followup]]
        self.assertEqual(len(cluster_events(results)), 1)
        self.assertIn("Barclays upgraded", results[1]["summary"])

    def test_ambiguous_missing_target_does_not_bridge_conflicting_changes(self):
        headlines = ["Barclays upgrades Nvidia to Buy with $210 price target", "Barclays upgrades Nvidia to Buy with $220 price target", "Barclays upgrades Nvidia to Buy"]
        results = [extract_article(article(t, id=str(i)), "NVDA") for i, t in enumerate(headlines)]
        self.assertEqual(len(cluster_events(results)), 3)

    def test_early_missing_target_does_not_bridge_later_conflicting_changes(self):
        headlines = ["Barclays upgrades Nvidia to Buy", "Barclays upgrades Nvidia to Buy with $210 price target", "Barclays upgrades Nvidia to Buy with $220 price target"]
        results = [extract_article(article(t, id=str(i)), "NVDA") for i, t in enumerate(headlines)]
        groups = cluster_events(results)
        self.assertEqual([group["article_ids"] for group in groups], [["0", "1"], ["2"]])

    def test_multi_company_ties_have_deterministic_order(self):
        source = article("Nvidia partners with Tesla", id="shared")
        results = [extract_article(source, symbol) for symbol in ["NVDA", "TSLA"]]
        self.assertEqual(cluster_events(results), cluster_events(list(reversed(results))))

    def test_different_brokers_dates_and_targets_stay_separate(self):
        title = "Morgan Stanley raises Nvidia price target to $210 from $190 on strong datacenter demand"
        sources = [article(title, id="a"), article(title.replace("Morgan Stanley", "Barclays"), id="b"),
                   article(title, id="c", day="2026-10-02"), article(title.replace("$210", "$220"), id="d")]
        self.assertEqual(len(cluster_events([extract_article(a, "NVDA") for a in sources])), 4)

    def test_daily_trading_reports_do_not_merge_across_days(self):
        title = "Nvidia shares rise as semiconductor stocks gain on strong demand"
        records = [extract_article(article(title, id=str(n), day=f"2026-10-0{n}"), "NVDA") for n in (1, 2)]
        self.assertEqual(len(cluster_events(records)), 2)

    def test_new_york_dates_not_utc_dates_delimit_trading_stories(self):
        title = "Nvidia shares rise as semiconductor stocks gain on strong demand"
        a, b = article(title, id="a"), article(title, id="b")
        a["created_at"], b["created_at"] = "2026-10-02T00:30:00Z", "2026-10-02T12:30:00Z"
        self.assertEqual(len(cluster_events([extract_article(a, "NVDA"), extract_article(b, "NVDA")])), 2)

    def test_different_explicit_ratings_do_not_merge(self):
        title = "Morgan Stanley reiterates Nvidia Buy rating on strong datacenter growth and demand outlook"
        a, b = article(title, id="a"), article(title.replace("Buy", "Hold"), id="b")
        self.assertEqual(len(cluster_events([extract_article(a, "NVDA"), extract_article(b, "NVDA")])), 2)

    def test_contract_termination_and_award_stay_separate(self):
        sources = [article("Tesla signs battery supply contract with Vendor Z", id="a"),
                   article("Tesla will not renew battery supply contract with Vendor Z", id="b")]
        self.assertEqual(len(cluster_events([extract_article(a, "TSLA") for a in sources])), 2)

    def test_unknown_time_and_tag_only_are_not_collapsed(self):
        records = []
        for n in range(2):
            source = article("Nvidia launches new datacenter chip platform", id=str(n))
            source["created_at"] = "2026-10-01"  # no timezone / intraday timestamp
            records.append(extract_article(source, "NVDA"))
        self.assertEqual(len(cluster_events(records)), 2)
        tagged = [extract_article(article("Other company signs supply deal", id=str(n)), "NVDA") for n in range(2)]
        self.assertEqual(len(cluster_events(tagged)), 2)

    def test_other_companies_never_merge(self):
        source = article("Nvidia and Tesla launch new joint research platform")
        self.assertEqual(len(cluster_events([extract_article(source, "NVDA"), extract_article(source, "TSLA")])), 2)

    def test_input_order_does_not_change_chronological_clusters(self):
        records = [extract_article(article("Nvidia launches a new datacenter chip platform", id=str(n), day=f"2026-10-0{n}"), "NVDA") for n in (1, 2, 3)]
        self.assertEqual(cluster_events(records), cluster_events(list(reversed(records))))


if __name__ == "__main__":
    unittest.main()
