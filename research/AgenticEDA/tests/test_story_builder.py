"""Tests for story builder plot curation."""
import pytest
from src.agent.knowledge_graph import KnowledgeGraph, KnowledgeNode


def _make_kg_with_investigation():
    """Create a KG with one investigation that has plot metadata."""
    kg = KnowledgeGraph()
    nid = kg.add_investigation(
        hypothesis_id="h1",
        hypothesis_title="CO bursts are heteroskedastic",
        finding="Variance increases 2.4x in high-CO regime (Levene p < 10^-56).",
        evidence_cells=["h1_cell_1", "h1_cell_2", "h1_cell_3"],
        plot_cells=["h1_cell_2", "h1_cell_3"],
        confidence=0.85,
        sub_findings=[],
        columns=["CO(GT)", "PT08.S1(CO)"],
        analysis_type="hypothesis_investigation",
    )
    # Simulate plot metadata stored by eda_agent
    node = kg.nodes[nid]
    node.metadata["plot_images"] = [
        {"cell_id": "h1_cell_2", "image_png": "base64data1"},
        {"cell_id": "h1_cell_3", "image_png": "base64data2"},
    ]
    node.metadata["cell_sources"] = {
        "h1_cell_1": "print(df['CO(GT)'].describe())",
        "h1_cell_2": "plt.scatter(df['CO(GT)'], df['PT08.S1(CO)'])\nplt.title('CO vs Sensor')\nplt.show()",
        "h1_cell_3": "plt.hist(residuals, bins=30)\nplt.title('Residual Distribution')\nplt.show()",
    }
    node.metadata["cell_outputs"] = {
        "h1_cell_1": "count    9357\nmean     2.15\nstd      1.45",
        "h1_cell_2": "[plot generated]",
        "h1_cell_3": "Levene p=1.996e-56\n[plot generated]",
    }
    return kg


def test_get_story_sections_includes_plot_metadata():
    kg = _make_kg_with_investigation()
    sections = kg.get_story_sections()
    inv_sections = [s for s in sections if s["type"] == "investigation"]
    assert len(inv_sections) == 1
    sec = inv_sections[0]
    assert "plot_metadata" in sec
    assert len(sec["plot_metadata"]) == 2
    meta0 = sec["plot_metadata"][0]
    assert "cell_id" in meta0
    assert "cell_source" in meta0
    assert "cell_output" in meta0


from src.reporting.story_builder import curate_section_plots, build_curated_story, build_story_abstract, _fallback_captions


def test_curate_section_plots_within_limit():
    """If plots <= max, generate captions for all."""
    section = {
        "title": "Test",
        "content": "Finding text.",
        "plot_metadata": [
            {"cell_id": "c1", "cell_source": "plt.scatter(x, y)\nplt.show()", "cell_output": "[plot]"},
            {"cell_id": "c2", "cell_source": "plt.hist(data)\nplt.show()", "cell_output": "[plot]"},
        ],
    }
    result = curate_section_plots(section, max_plots=3)
    assert len(result) <= 3
    assert len(result) >= 1  # at least fallback
    for item in result:
        assert "cell_id" in item
        assert "caption" in item


def test_curate_section_plots_empty_metadata():
    """If no plot metadata, return empty list."""
    section = {"title": "Data Loading", "content": "Loaded 9357 rows.", "plot_metadata": []}
    result = curate_section_plots(section, max_plots=3)
    assert result == []


def test_fallback_captions_extracts_title():
    """Fallback should extract plt.title from source code."""
    section = {"title": "Hypothesis A"}
    plots = [
        {"cell_id": "c1", "cell_source": "plt.scatter(x, y)\nplt.title('CO vs Sensor Response')\nplt.show()", "cell_output": ""},
    ]
    result = _fallback_captions(section, plots)
    assert len(result) == 1
    assert result[0]["caption"] == "CO vs Sensor Response"


def test_fallback_captions_chart_type():
    """Fallback should detect chart type from source."""
    section = {"title": "My Analysis"}
    plots = [
        {"cell_id": "c1", "cell_source": "plt.hist(data, bins=30)\nplt.show()", "cell_output": ""},
    ]
    result = _fallback_captions(section, plots)
    assert "Distribution" in result[0]["caption"]


def test_build_curated_story_structure():
    """Curated story should have abstract and curated sections."""
    sections = [{
        "title": "Test",
        "content": "Finding.",
        "plot_cell_ids": ["c1"],
        "type": "investigation",
        "plot_metadata": [
            {"cell_id": "c1", "cell_source": "plt.plot(x)\nplt.show()", "cell_output": "[plot]"},
        ],
    }]
    story = build_curated_story(sections, "This is the summary. It has findings.", max_plots_per_section=2)
    assert "abstract" in story
    assert "sections" in story
    assert "executive_summary" in story
    assert "generated_at" in story
