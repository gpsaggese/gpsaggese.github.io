"""PatchTST-style baseline smoke tests; not claims of financial performance."""
import numpy as np
import pandas as pd
import torch
import pytest

from benchmark_models import fit_deep, make_common_frame, split_chronologically, to_seq
from patchtst_model import PatchTSTReturnRegressor


def test_patchtst_forward_and_backward():
    torch.manual_seed(42)
    model = PatchTSTReturnRegressor(context_length=30, patch_len=6, stride=3,
                                     hidden=32, heads=4, layers=2)
    seq = torch.randn(7, 30)
    aux = torch.randn(7, 2)
    pred = model(seq, aux)
    assert pred.shape == (7,)
    assert torch.isfinite(pred).all()
    pred.square().mean().backward()
    assert model.patch_embedding.weight.grad is not None
    assert model.n_patches == 9


def test_patchtst_rejects_bad_shapes():
    m = PatchTSTReturnRegressor(context_length=30)
    with pytest.raises(ValueError):
        m(torch.zeros(2, 29), torch.zeros(2, 2))
    with pytest.raises(ValueError):
        m(torch.zeros(2, 30), torch.zeros(2, 3))


def test_patchtst_quick_real_pipeline_synthetic():
    # Synthetic nonconstant price returns are for integration verification only.
    torch.set_num_threads(min(2, torch.get_num_threads()))
    rng = np.random.default_rng(41)
    returns = rng.normal(0.0002, 0.012, size=260)
    prices = 55.0 * np.exp(np.cumsum(returns))
    raw = pd.DataFrame({'date': pd.bdate_range('2020-01-01', periods=260),
                        'price': prices})
    frame = make_common_frame(raw)
    assert (frame.feature_asof_date < frame.date).all()
    parts = split_chronologically(frame, gap=5)
    pred, config = fit_deep('patchtst', parts['train'], parts['validation'],
                            parts['test'], torch.device('cpu'), quick=True)
    assert pred.shape == (len(parts['test']),)
    assert np.isfinite(pred).all()
    assert config['selected_config']['patch_length'] == 6
    assert config['selected_config']['attention_heads'] == 4
    assert len(to_seq(parts['test'][0:1])[0]) == 30
