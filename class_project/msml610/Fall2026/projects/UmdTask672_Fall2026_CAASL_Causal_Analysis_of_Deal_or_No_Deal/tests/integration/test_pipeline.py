"""Validate sample counts against the checked-in ZBW data."""

from pathlib import Path
import numpy as np
import pytest

from analysis import run_analysis
from build_dataset import build_dataset
from cleaner import clean
from features import build_features
from load_zbw import load_zbw


pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def pipeline_stages():
    project_root = Path(__file__).resolve().parents[2]
    raw = load_zbw(project_root / "data/raw/zbw/deal-data.txt")
    cleaned = clean(raw)
    reconstructed, mismatches = build_dataset(cleaned)
    offers = build_features(reconstructed)
    return raw, cleaned, reconstructed, mismatches, offers


def test_cleaning_and_reconstruction_counts(pipeline_stages):
    raw, cleaned, reconstructed, mismatches, offers = pipeline_stages
    assert len(raw) == 4664
    assert cleaned["show_id"].nunique() == 311
    assert reconstructed["show_id"].nunique() == 310
    assert set(mismatches["show_id"]) == {336}
    assert len(offers) == 2136
    assert offers["show_id"].nunique() == 291
    assert offers["decision"].notna().all()
    first_deal = reconstructed.loc[reconstructed["decision"].eq(1)].groupby(
        "show_id"
    )["round"].min()
    deal_round = offers["show_id"].map(first_deal)
    assert (deal_round.isna() | offers["round"].le(deal_round)).all()
    assert offers["n_remaining"].gt(0).all()
    for values in reconstructed.groupby("show_id")["n_remaining"].apply(list):
        assert values == sorted(values, reverse=True)


def test_no_contaminated_shows_in_final_output(pipeline_stages):
    raw, cleaned, _, _, offers = pipeline_stages
    contaminated = set(raw.loc[raw["special_show"].isin([1, 2, 3]), "show_id"])
    assert cleaned["special_show"].eq(0).all()
    assert set(offers["show_id"]).isdisjoint(contaminated)


def test_normalized_luck_coverage(pipeline_stages):
    _, _, reconstructed, _, offers = pipeline_stages
    assert offers["luck_undefined"].equals(offers["normalized_luck"].isna())
    model_df = offers.loc[~offers["luck_undefined"]]
    assert len(model_df) == 1845
    assert model_df["normalized_luck"].notna().all()
    assert np.isfinite(model_df["normalized_luck"]).all()
    # Games absent from the offer sample have no recorded banker offers.
    excluded = set(reconstructed["show_id"]) - set(offers["show_id"])
    assert len(excluded) == 19
    assert reconstructed.loc[reconstructed["show_id"].isin(excluded), "bank_offer"].isna().all()


@pytest.mark.filterwarnings("ignore:normalized_luck is positive:RuntimeWarning")
def test_real_data_analysis_converges(pipeline_stages):
    result = run_analysis(pipeline_stages[-1])
    assert result.mle_retvals["converged"]
    assert result.cov_type == "cluster"
    assert result.nobs == 1845
    assert np.isfinite(result.params).all()
    assert np.isfinite(result.bse).all()
