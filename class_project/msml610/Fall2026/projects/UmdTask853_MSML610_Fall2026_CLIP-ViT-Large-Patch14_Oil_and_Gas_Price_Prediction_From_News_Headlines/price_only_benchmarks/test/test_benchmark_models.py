import numpy as np
import pandas as pd
from benchmark_models import make_common_frame, regression_metrics, SERIES, tabular_grid
from forecast_utils import split_chronologically


def test_target_and_features_come_from_previous_observations():
    dates = pd.bdate_range('2020-01-01',periods=100)
    # Deterministically changing daily returns: check that lag_1 is previous target.
    returns = 0.001+0.0002*np.arange(100)
    prices = 40*np.exp(np.cumsum(returns))
    f=make_common_frame(pd.DataFrame({'date':dates, 'price':prices}))
    assert len(f)>50
    original=np.log(prices[1:]/prices[:-1])
    for _,row in f.iterrows():
        ix=dates.get_loc(row['date'])
        assert row['feature_asof_date']==dates[ix-1]
        assert np.isclose(row['target_return'],original[ix-1])
        assert np.isclose(row['return_lag_1'],original[ix-2])


def test_negative_wti_gap_not_bridged():
    d=pd.bdate_range('2019-01-01',periods=100)
    p=40*np.exp(np.cumsum(0.001*np.ones(100)))
    p[60]=-10
    f=make_common_frame(pd.DataFrame({'date':d,'price':p}))
    assert not d[60] in set(f.date)
    assert not d[61] in set(f.date)
    assert d[62] not in set(f.date)  # The historical lags include invalid returns.


def test_metrics_and_models():
    y=np.array([0.02,-0.01,0.03,-0.02]);p=np.array([0.01,-0.02,0.01,-0.01])
    m=regression_metrics(y,p,0.0)
    assert m['rmse']>0 and 0<=m['directional_accuracy']<=1
    assert set(SERIES)=={'wti','brent','henry_hub'}
    for model in ['ridge','elasticnet','histgb','xgboost','lightgbm']:
        est,grid=tabular_grid(model,quick=True)
        assert grid and est is not None
