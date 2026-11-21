"""Heading line ranges, which the Parse step colours the record by."""
from adrqa.parser import section_spans
from test_offline_units import RECORD


def test_section_spans_give_line_ranges_and_nest_subsections():
    spans = section_spans(RECORD)
    by_name = {sp["name"]: sp for sp in spans}
    assert by_name["Some Title"]["level"] == 1 and by_name["Some Title"]["end"] == len(RECORD.splitlines())
    assert (by_name["Decision"]["field"], by_name["Rationale"]["field"], by_name["Approach"]["field"]) == ("decision", "rationale", "approach")
    # "# not a heading" sits inside a code fence, so it is not a heading
    assert "not a heading" not in by_name
    assert by_name["Decision"]["end"] == by_name["Rationale"]["start"]
    sub = by_name["Sub heading"]
    assert sub["level"] == 3 and by_name["Approach"]["start"] < sub["start"] and sub["end"] <= by_name["Approach"]["end"]
    assert sub["field"] is None
