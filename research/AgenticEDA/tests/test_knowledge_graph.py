"""Tests for the redesigned knowledge graph."""
import pytest
from src.agent.knowledge_graph import KnowledgeGraph, KnowledgeNode, KGEdge


@pytest.fixture
def kg():
    return KnowledgeGraph()


@pytest.fixture
def populated_kg(kg):
    """KG with some facts and an investigation."""
    kg.add_fact("Dataset has 1000 rows", "Data Loading", cell_id="c1")
    kg.add_fact("Column age has 5% nulls", "Data Cleaning", cell_id="c2")
    kg.add_investigation(
        hypothesis_id="h1",
        hypothesis_title="Age vs Income",
        finding="Strong positive correlation between age and income",
        evidence_cells=["c3"],
        plot_cells=["p1"],
        confidence=0.85,
        sub_findings=[
            {"finding": "Pearson r=0.72", "cell_ids": ["c4"], "plot_cells": ["p2"], "confidence": 0.9},
            {"finding": "Holds for ages 25-65", "cell_ids": ["c5"], "confidence": 0.8},
        ],
    )
    return kg


class TestAddFact:
    def test_returns_id(self, kg):
        nid = kg.add_fact("some fact", "Data Loading")
        assert isinstance(nid, str)
        assert nid.startswith("fact_")

    def test_node_stored(self, kg):
        nid = kg.add_fact("some fact", "Data Loading", cell_id="c1")
        node = kg.nodes[nid]
        assert node.type == "fact"
        assert node.text == "some fact"
        assert node.phase == "Data Loading"
        assert "c1" in node.cell_ids

    def test_metadata_columns(self, kg):
        nid = kg.add_fact("age distribution", "Univariate Analysis", columns=["age"])
        node = kg.nodes[nid]
        assert node.metadata.get("columns") == ["age"]


class TestAddInvestigation:
    def test_creates_conclusion_and_evidence(self, kg):
        nid = kg.add_investigation(
            hypothesis_id="h1",
            hypothesis_title="Test",
            finding="Main finding",
            evidence_cells=["c1"],
            plot_cells=["p1"],
            confidence=0.8,
            sub_findings=[{"finding": "sub1", "cell_ids": ["c2"]}],
        )
        conclusion = kg.nodes[nid]
        assert conclusion.type == "conclusion"
        assert len(conclusion.children) == 1
        child = kg.nodes[conclusion.children[0]]
        assert child.type == "evidence"
        assert child.text == "sub1"

    def test_supports_edges_created(self, kg):
        nid = kg.add_investigation(
            hypothesis_id="h1",
            hypothesis_title="Test",
            finding="Main finding",
            evidence_cells=["c1"],
            plot_cells=["p1"],
            sub_findings=[
                {"finding": "sub1", "cell_ids": ["c2"]},
                {"finding": "sub2", "cell_ids": ["c3"]},
            ],
        )
        supports_edges = [e for e in kg.edges if e.type == "supports" and e.target_id == nid]
        assert len(supports_edges) == 2

    def test_columns_metadata(self, kg):
        nid = kg.add_investigation(
            hypothesis_id="h1",
            hypothesis_title="Test",
            finding="Main",
            evidence_cells=[],
            plot_cells=[],
            columns=["age", "income"],
        )
        assert kg.nodes[nid].metadata["columns"] == ["age", "income"]

    def test_analysis_type_metadata(self, kg):
        nid = kg.add_investigation(
            hypothesis_id="h1",
            hypothesis_title="Test",
            finding="Main",
            evidence_cells=[],
            plot_cells=[],
            analysis_type="correlation",
        )
        assert kg.nodes[nid].metadata["analysis_type"] == "correlation"


class TestGetStorySections:
    def test_returns_initial_and_investigation(self, populated_kg):
        sections = populated_kg.get_story_sections()
        types = [s["type"] for s in sections]
        assert "initial" in types
        assert "investigation" in types

    def test_initial_section_content(self, populated_kg):
        sections = populated_kg.get_story_sections()
        initial = [s for s in sections if s["type"] == "initial"]
        assert len(initial) >= 1

    def test_investigation_section_has_evidence(self, populated_kg):
        sections = populated_kg.get_story_sections()
        inv = [s for s in sections if s["type"] == "investigation"]
        assert len(inv) == 1
        assert "Supporting evidence" in inv[0]["content"]


class TestGetTopConclusions:
    def test_returns_strings(self, populated_kg):
        conclusions = populated_kg.get_top_conclusions()
        assert all(isinstance(c, str) for c in conclusions)
        assert len(conclusions) >= 1

    def test_limited_by_n(self, populated_kg):
        conclusions = populated_kg.get_top_conclusions(n=1)
        assert len(conclusions) <= 1


class TestAddEdge:
    def test_stores_edge(self, kg):
        kg.add_fact("a", "Data Loading")
        kg.add_fact("b", "Data Loading")
        edge = KGEdge(source_id="fact_1", target_id="fact_2", type="supports")
        kg.add_edge(edge)
        assert len(kg.edges) == 1
        assert kg.edges[0].type == "supports"


class TestGetEvidenceChain:
    def test_follows_supports_edges(self, populated_kg):
        # Find the conclusion node
        conclusion_ids = [nid for nid, n in populated_kg.nodes.items() if n.type == "conclusion"]
        assert len(conclusion_ids) == 1
        chain = populated_kg.get_evidence_chain(conclusion_ids[0])
        # Should include the evidence nodes that support the conclusion
        assert len(chain) >= 2  # at least the 2 sub_findings

    def test_empty_for_no_edges(self, kg):
        nid = kg.add_fact("orphan", "Data Loading")
        chain = kg.get_evidence_chain(nid)
        assert chain == []


class TestQueryByColumns:
    def test_finds_by_metadata_columns(self, kg):
        kg.add_fact("age stats", "Univariate Analysis", columns=["age", "income"])
        results = kg.query_by_columns(["age"])
        assert len(results) >= 1
        assert any("age" in r.text for r in results)

    def test_finds_by_text_mention(self, kg):
        kg.add_fact("The age column is skewed", "Univariate Analysis")
        results = kg.query_by_columns(["age"])
        assert len(results) >= 1


class TestQueryByType:
    def test_filters_by_type(self, populated_kg):
        facts = populated_kg.query_by_type("fact")
        assert all(n.type == "fact" for n in facts)
        assert len(facts) == 2

    def test_only_latest(self, kg):
        nid = kg.add_fact("old", "Data Loading")
        kg.nodes[nid].is_latest = False
        results = kg.query_by_type("fact")
        assert len(results) == 0


class TestFindDuplicateInvestigation:
    def test_detects_exact_match(self, kg):
        kg.add_investigation(
            hypothesis_id="h1",
            hypothesis_title="Test",
            finding="result",
            evidence_cells=[],
            plot_cells=[],
            columns=["age", "income"],
            analysis_type="correlation",
        )
        dup = kg.find_duplicate_investigation(["age", "income"], "correlation")
        assert dup is not None

    def test_no_false_positive(self, kg):
        kg.add_investigation(
            hypothesis_id="h1",
            hypothesis_title="Test",
            finding="result",
            evidence_cells=[],
            plot_cells=[],
            columns=["age", "income"],
            analysis_type="correlation",
        )
        dup = kg.find_duplicate_investigation(["age", "salary"], "regression")
        assert dup is None


class TestFindSimilarHypothesis:
    def test_finds_column_overlap(self, kg):
        kg.add_investigation(
            hypothesis_id="h1",
            hypothesis_title="Age Income Corr",
            finding="Positive correlation",
            evidence_cells=[],
            plot_cells=[],
            columns=["age", "income", "education"],
            analysis_type="correlation",
        )
        result = kg.find_similar_hypothesis("Relationship between age and income", threshold=0.3)
        assert result is not None

    def test_no_match_below_threshold(self, kg):
        kg.add_investigation(
            hypothesis_id="h1",
            hypothesis_title="Weather patterns",
            finding="Rain correlates with humidity",
            evidence_cells=[],
            plot_cells=[],
            columns=["rain", "humidity"],
        )
        result = kg.find_similar_hypothesis("age vs income analysis", threshold=0.5)
        assert result is None


class TestReinforce:
    def test_increases_confidence(self, kg):
        nid = kg.add_fact("x", "Data Loading")
        kg.nodes[nid].confidence = 0.5
        old_conf = kg.nodes[nid].confidence
        kg.reinforce(nid)
        assert kg.nodes[nid].confidence > old_conf

    def test_caps_at_099(self, kg):
        nid = kg.add_fact("x", "Data Loading")
        kg.nodes[nid].confidence = 0.98
        for _ in range(20):
            kg.reinforce(nid)
        assert kg.nodes[nid].confidence <= 0.99


class TestSupersede:
    def test_marks_old_superseded(self, kg):
        old = kg.add_fact("old fact", "Data Loading")
        new = kg.add_fact("new fact", "Data Loading")
        kg.supersede(old, new)
        assert kg.nodes[old].is_latest is False
        assert kg.nodes[old].superseded_by == new

    def test_adds_supersedes_edge(self, kg):
        old = kg.add_fact("old", "Data Loading")
        new = kg.add_fact("new", "Data Loading")
        kg.supersede(old, new)
        edges = [e for e in kg.edges if e.type == "supersedes"]
        assert len(edges) == 1
        assert edges[0].source_id == new
        assert edges[0].target_id == old


class TestGetContextForHypothesisGeneration:
    def test_includes_facts_and_conclusions(self, populated_kg):
        ctx = populated_kg.get_context_for_hypothesis_generation()
        assert "1000 rows" in ctx
        assert "correlation" in ctx.lower()


class TestGetContextForChat:
    def test_returns_relevant_nodes(self, populated_kg):
        ctx = populated_kg.get_context_for_chat("What about age and income correlation?")
        assert "age" in ctx.lower() or "income" in ctx.lower()

    def test_handles_no_matches(self, kg):
        ctx = kg.get_context_for_chat("anything")
        assert isinstance(ctx, str)


class TestToDictFromDict:
    def test_roundtrip_preserves_nodes(self, populated_kg):
        data = populated_kg.to_dict()
        kg2 = KnowledgeGraph.from_dict(data)
        assert set(kg2.nodes.keys()) == set(populated_kg.nodes.keys())
        for nid in populated_kg.nodes:
            assert kg2.nodes[nid].text == populated_kg.nodes[nid].text

    def test_roundtrip_preserves_edges(self, populated_kg):
        data = populated_kg.to_dict()
        kg2 = KnowledgeGraph.from_dict(data)
        assert len(kg2.edges) == len(populated_kg.edges)
        for e1, e2 in zip(populated_kg.edges, kg2.edges):
            assert e1.source_id == e2.source_id
            assert e1.target_id == e2.target_id
            assert e1.type == e2.type
