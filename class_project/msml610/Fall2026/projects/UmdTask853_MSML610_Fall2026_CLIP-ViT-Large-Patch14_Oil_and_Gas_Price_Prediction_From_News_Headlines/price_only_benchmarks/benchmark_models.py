"""Leakage-aware energy *price-only* regression benchmarks.

Supports WTI, Brent, Henry Hub. All models use past-price information only.
No news, FinBERT, or CLIP is used in these baselines.
Produces test RMSE, MAE, OOS R2, sign accuracy, selection params and predictions.

Run from this script's directory (or copy next to forecast_utils.py).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecast_utils import (PRICE_FEATURES, make_price_features,
                            read_fred_csv, split_chronologically)

SERIES = {"wti": "DCOILWTICO", "brent": "DCOILBRENTEU", "henry_hub": "DHHNGSP"}
RETURN_LAGS = 30
FEATURES = [f"return_lag_{k}" for k in range(1, RETURN_LAGS + 1)] + ["momentum_5", "volatility_20"]
SEED = 42


def seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_common_frame(raw: pd.DataFrame, *, lags: int = RETURN_LAGS) -> pd.DataFrame:
    """Return for observed target date t, with all predictors <= previous quote.

    Gap around nonpositive price: never compute a return across the invalid quote.
    Common 30-return history ensures the same eligible dates for every model.
    """
    if lags < 20:
        raise ValueError("Use at least 20 historical returns")
    x = raw[["date", "price"]].sort_values("date").copy().reset_index(drop=True)
    x["price"] = pd.to_numeric(x.price, errors="coerce")
    x = x.loc[x.price.notna()].reset_index(drop=True)
    previous = x.price.shift(1)
    x["previous_price"] = previous
    valid_pair = x.price.gt(0) & previous.gt(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        x["log_return"] = np.log(x.price / previous).where(valid_pair)
    x["feature_asof_date"] = x.date.shift(1)
    x["target_return"] = x.log_return
    for k in range(1, lags + 1):
        x[f"return_lag_{k}"] = x.log_return.shift(k)
    x["momentum_5"] = x.log_return.shift(1).rolling(5, min_periods=5).sum()
    x["volatility_20"] = x.log_return.shift(1).rolling(20, min_periods=20).std()
    cols = [f"return_lag_{i}" for i in range(1, lags+1)] + ["momentum_5", "volatility_20"]
    out = x.dropna(subset=cols+["target_return", "feature_asof_date"]).copy()
    out = out.reset_index(drop=True)
    if out.empty or not (out.feature_asof_date < out.date).all():
        raise ValueError("Empty or leaking price frame")
    if not np.isfinite(out[cols+["target_return"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite feature or target")
    return out


def abc_date_window(abc_csv: str, raw: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Use complete ABC source coverage, not keyword-match coverage.

    Include at most the first target-price date STRICTLY after ABC's last date.
    """
    dates = pd.read_csv(abc_csv, usecols=["publish_date"], dtype=str).publish_date
    parsed = pd.to_datetime(dates.str.replace(r"\.0$", "", regex=True), format="%Y%m%d", errors="coerce")
    parsed = parsed.dropna()
    if parsed.empty:
        raise ValueError("ABC dataset has no parseable dates")
    valid = pd.DatetimeIndex(raw.loc[raw.price.gt(0), "date"].unique()).sort_values()
    lo = valid.searchsorted(parsed.min(), side="right")
    hi = valid.searchsorted(parsed.max(), side="right")
    if lo >= len(valid) or hi >= len(valid):
        raise ValueError("FRED price series does not cover ABC publication end")
    return valid[lo], valid[hi]


def regression_metrics(y, pred, reference):
    y = np.asarray(y, dtype=float); pred = np.asarray(pred, dtype=float)
    assert y.shape == pred.shape and y.ndim == 1
    mse = np.mean((y-pred)**2)
    baseline_mse = np.mean((y-reference)**2)
    is_direction = (np.sign(y) != 0) & (np.sign(pred) != 0)
    return {"rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(y-pred))),
            "oos_r2": float(1-mse/baseline_mse) if baseline_mse > 0 else float("nan"),
            "directional_accuracy": float(np.mean(np.sign(y[is_direction]) == np.sign(pred[is_direction]))) if np.any(is_direction) else float("nan")}


def tabular_grid(name, quick=False):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [1.] if quick else [.1,1.,10.,100.,1000.]}
    if name == "elasticnet":
        return make_pipeline(StandardScaler(), ElasticNet(max_iter=20000, random_state=SEED)), {
            "elasticnet__alpha": [.001] if quick else [.0001,.001,.01],
            "elasticnet__l1_ratio": [.5] if quick else [.1,.5,.9]}
    if name == "histgb":
        return HistGradientBoostingRegressor(random_state=SEED, early_stopping=False), {
            "max_iter": [100] if quick else [100, 250],
            "max_leaf_nodes": [7] if quick else [7, 15],
            "learning_rate": [.03] if quick else [.03, .07],
            "l2_regularization": [1.] if quick else [1., 10.]}
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("Install lightgbm: python -m pip install lightgbm") from exc
        return LGBMRegressor(random_state=SEED, n_jobs=2, verbosity=-1,
                             subsample=.8, colsample_bytree=.8), {
            "n_estimators": [120] if quick else [100, 250],
            "num_leaves": [7] if quick else [7, 15],
            "learning_rate": [.03] if quick else [.03,.07],
            "reg_lambda": [10.] if quick else [1.,10.]}
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("Install xgboost: python -m pip install xgboost") from exc
        return XGBRegressor(tree_method="hist", objective="reg:squarederror", n_jobs=2,
                            random_state=SEED, subsample=.8, colsample_bytree=.8), {
            "n_estimators": [150] if quick else [100,250],
            "max_depth": [2] if quick else [2,3],
            "learning_rate": [.03] if quick else [.03,.07],
            "reg_lambda": [10.] if quick else [1.,10.]}
    raise ValueError(f"Unknown tabular model: {name}")


def fit_tabular(name, train, val, test, quick=False):
    estimator, grid = tabular_grid(name, quick)
    cv = TimeSeriesSplit(n_splits=3, gap=5)
    tuned = GridSearchCV(estimator, grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=1,
                         error_score="raise", refit=True)
    tuned.fit(train[FEATURES], train.target_return)
    # Use validation only to report model diagnostics; model selection is within training CV.
    val_pred = tuned.predict(val[FEATURES])
    # Once selected, refit the *same* configuration on train+validation (excluding gaps).
    trainval = pd.concat([train, val], ignore_index=True)
    fitted = tuned.best_estimator_
    fitted.fit(trainval[FEATURES], trainval.target_return)
    test_pred = fitted.predict(test[FEATURES])
    params = {"search": tuned.best_params_, "inner_cv_best_rmse": float(np.sqrt(-tuned.best_score_)),
              "validation_rmse_pre_refit": float(np.sqrt(mean_squared_error(val.target_return,val_pred)))}
    return test_pred, params


def to_seq(frame):
    # Sequence oldest->newest. Target date's return is never an input.
    return frame[[f"return_lag_{k}" for k in range(RETURN_LAGS, 0, -1)]].to_numpy(dtype=np.float32)


def fit_deep(kind, train, val, test, device, quick=False):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class PriceSequenceModel(nn.Module):
        def __init__(self, kind, hidden):
            super().__init__()
            self.kind = kind
            if kind == "tcn":
                # Causal dilated convolutions: receptive field 31 observations.
                self.dilations = (1, 2, 4, 8)
                self.convs = nn.ModuleList([
                    nn.Conv1d(1 if i == 0 else hidden, hidden,
                              kernel_size=3, dilation=d, padding=2*d)
                    for i, d in enumerate(self.dilations)
                ])
                self.dropout = nn.Dropout(p=.10)
            elif kind == "patchtst":
                from patchtst_model import PatchTSTReturnRegressor
                self.patchtst = PatchTSTReturnRegressor(
                    context_length=RETURN_LAGS, patch_len=6, stride=3,
                    hidden=hidden, heads=4, layers=2, dropout=0.10,
                )
            else:
                cell = nn.LSTM if kind == "lstm" else nn.GRU
                self.encoder = cell(1, hidden, batch_first=True)
            if kind != "patchtst":
                self.head = nn.Sequential(nn.Linear(hidden+2, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))

        def forward(self, seq, aux):
            if self.kind == "patchtst":
                return self.patchtst(seq, aux)
            if self.kind == "tcn":
                x = seq.unsqueeze(1)
                for conv, d in zip(self.convs, self.dilations):
                    # Trim padded future positions: only past history contributes.
                    z = self.dropout(torch.relu(conv(x)[:, :, :-(2*d)]))
                    x = z + x if x.shape[1] == z.shape[1] else z
                h = x[:, :, -1]
            else:
                output, _ = self.encoder(seq.unsqueeze(-1))
                h = output[:, -1, :]
            return self.head(torch.cat([h, aux], dim=1)).squeeze(1)

    def tensors(parts, fit_train):
        full_series = to_seq(fit_train).reshape(-1, 1)
        seq_scaler = StandardScaler().fit(full_series)
        aux_scaler = StandardScaler().fit(fit_train[["momentum_5","volatility_20"]])
        ys = fit_train.target_return.to_numpy()
        mean, std = float(ys.mean()), max(float(ys.std()), 1e-5)
        data = []
        for p in parts:
            a = seq_scaler.transform(to_seq(p).reshape(-1,1)).reshape(-1, RETURN_LAGS).astype(np.float32)
            b = aux_scaler.transform(p[["momentum_5","volatility_20"]]).astype(np.float32)
            c = ((p.target_return.to_numpy()-mean)/std).astype(np.float32)
            data.append((torch.from_numpy(a), torch.from_numpy(b),torch.from_numpy(c)))
        return data, mean, std

    def train_loop(fit_data, val_data, cfg, fixed_epochs=None):
        seeds(SEED)
        m = PriceSequenceModel(kind, cfg["hidden"]).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        loader = DataLoader(TensorDataset(*fit_data), batch_size=cfg["batch_size"], shuffle=True)
        best = float("inf"); best_epoch = 0; best_state = None; stale = 0
        n_epochs = fixed_epochs or cfg["epochs"]
        for epoch in range(n_epochs):
            m.train()
            for seq, aux, y in loader:
                seq,aux,y = seq.to(device), aux.to(device), y.to(device)
                opt.zero_grad(); loss = torch.mean((m(seq,aux)-y)**2)
                loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.)
                opt.step()
            if val_data is not None:
                m.eval()
                with torch.no_grad():
                    s,a,y = (z.to(device) for z in val_data)
                    score = float(torch.mean((m(s,a)-y)**2).item())
                if score < best-1e-6:
                    best, best_epoch, best_state, stale = score, epoch+1, {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}, 0
                else:
                    stale += 1
                if stale >= 5:
                    break
        if best_state is not None:
            m.load_state_dict(best_state)
        return m, best, best_epoch or n_epochs

    if kind == "patchtst":
        # Small two-layer patch-transformer; keep training compute modest
        # for ~3k-6k labelled training windows per commodity.
        tuning = [(32,1e-3)] if quick else [(32,1e-3), (64,3e-4)]
        candidates = [{"hidden":h,"lr":lr,"weight_decay":1e-3,
                       "batch_size":64,"epochs":2 if quick else 40,
                       "patch_length":6,"patch_stride":3,"transformer_layers":2,
                       "attention_heads":4,"dropout":.10}
                      for h,lr in tuning]
    else:
        candidates = [{"hidden":h,"lr":lr,"weight_decay":1e-3,"batch_size":64,
                       "epochs":6 if quick else 25} for h,lr in ([(16,1e-3)] if quick else [(16,1e-3),(32,3e-4)])]
    prepared, mu, sigma = tensors([train,val], train)
    results = []
    for cfg in candidates:
        model, loss, ep = train_loop(prepared[0], prepared[1], cfg)
        results.append((loss, cfg, ep))
    score, cfg, ep = min(results, key=lambda x:x[0])
    trainval = pd.concat([train,val], ignore_index=True)
    datas, mu, sigma = tensors([trainval,test], trainval)
    final, _,_ = train_loop(datas[0], None, cfg, fixed_epochs=ep)
    final.eval()
    with torch.no_grad():
        s,a,_=datas[1]; p=final(s.to(device),a.to(device)).cpu().numpy() * sigma + mu
    return p, {"selected_config":cfg,"selected_epochs":ep,"validation_standardized_mse":score,"device":str(device)}


def fit_contrastive(train,val,test,device,quick=False):
    """Simple augmentation+InfoNCE price encoder, NOT the full TS2Vec algorithm."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader,TensorDataset
    class Encoder(nn.Module):
        def __init__(self, hidden=32):
            super().__init__()
            self.gru=nn.GRU(1,hidden,batch_first=True)
            self.proj=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        def encode(self,x):
            return self.gru(x.unsqueeze(-1))[0][:,-1,:]
        def forward(self,x):
            return self.proj(self.encode(x))
    def pretrain(fit_train, n_epochs):
        seeds(SEED)
        scaler=StandardScaler().fit(to_seq(fit_train).reshape(-1,1))
        seq=scaler.transform(to_seq(fit_train).reshape(-1,1)).reshape(-1,RETURN_LAGS).astype(np.float32)
        net=Encoder().to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
        loader=DataLoader(TensorDataset(torch.from_numpy(seq)),batch_size=128,shuffle=True,drop_last=False)
        for _ in range(n_epochs):
            net.train()
            for (batch,) in loader:
                if len(batch)<2: continue
                x=batch.to(device)
                views=[]
                for _ in range(2):
                    mask=(torch.rand_like(x)>.10).float()
                    views.append(x*mask+0.02*torch.randn_like(x))
                z1=nn.functional.normalize(net(views[0]),dim=1)
                z2=nn.functional.normalize(net(views[1]),dim=1)
                z=torch.cat([z1,z2],dim=0)
                sim=(z @ z.T)/.2
                sim.fill_diagonal_(float("-inf"))
                n=len(x); labels=torch.arange(2*n,device=device)
                labels=(labels+n)%(2*n)
                loss=nn.functional.cross_entropy(sim,labels)
                opt.zero_grad();loss.backward();opt.step()
        net.eval()
        return net,scaler
    def feat(net,scaler,frame):
        seq=scaler.transform(to_seq(frame).reshape(-1,1)).reshape(-1,RETURN_LAGS)
        with torch.no_grad():
            emb=net.encode(torch.tensor(seq,dtype=torch.float32,device=device)).cpu().numpy()
        # Only add already available 30-lookback features; never target-date price.
        return np.column_stack([emb,frame[["momentum_5","volatility_20"]].to_numpy()])
    epochs=2 if quick else 18
    net,sc=pretrain(train,epochs)
    a,b=feat(net,sc,train),feat(net,sc,val)
    best=None
    for alpha in ([10.] if quick else [.1,1.,10.,100.,1000.]):
        model=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(a,train.target_return)
        score=float(np.sqrt(mean_squared_error(val.target_return,model.predict(b))))
        if best is None or score < best[0]: best=(score,alpha)
    trainval=pd.concat([train,val],ignore_index=True)
    net,sc=pretrain(trainval,epochs)
    model=make_pipeline(StandardScaler(),Ridge(alpha=best[1])).fit(feat(net,sc,trainval),trainval.target_return)
    pred=model.predict(feat(net,sc,test))
    return pred,{"method":"augmentation InfoNCE + frozen GRU embeddings + Ridge (not TS2Vec)",
                 "pretraining_epochs":epochs,"contrastive_temperature":.2,
                 "ridge_alpha":best[1],"validation_rmse":best[0],"device":str(device)}


def evaluate_one(asset,filename,output,models,period,abc_csv,device,quick):
    raw=read_fred_csv(filename,SERIES[asset]); frame=make_common_frame(raw)
    period_start,period_end=frame.date.min(),frame.date.max()
    if period == "abc":
        if not abc_csv: raise ValueError("--abc-csv required for --period abc")
        period_start,period_end=abc_date_window(abc_csv,raw)
        frame=frame.loc[frame.date.between(period_start,period_end)].reset_index(drop=True)
    parts=split_chronologically(frame,gap=5)
    train,val,test=(parts[k] for k in ("train","validation","test"))
    trainval=pd.concat([train,val],ignore_index=True)
    # The zero, mean models use the same available history used by refitted models.
    train_mean=float(trainval.target_return.mean())
    actual=test.target_return.to_numpy(dtype=float)
    predictions={"zero_return":np.zeros(len(test)),"historical_mean":np.full(len(test),train_mean)}
    settings={"input":str(filename),"raw_rows":len(raw),"raw_start":str(raw.date.min().date()),
              "raw_end":str(raw.date.max().date()),"missing_prices":int(raw.price.isna().sum()),
              "nonpositive_prices":int(raw.price.le(0).sum()),"eligible_rows":len(frame),
              "train_n":len(train),"validation_n":len(val),"test_n":len(test),
              "target": "log price change vs previous valid observed quote",
              "period":period,"from":str(frame.date.min().date()),"to":str(frame.date.max().date()),
              "test_from":str(test.date.min().date()),"test_to":str(test.date.max().date()),
              "feature_asof_max_before_target":bool((frame.feature_asof_date<frame.date).all()),
              "model_parameters":{}}
    for name in models:
        if name in ("zero_return","historical_mean"):continue
        print(f"[{asset}] Fitting {name} on {len(train):,} train + {len(val):,} val rows",flush=True)
        if name in ("ridge","elasticnet","histgb","xgboost","lightgbm"):
            pred,param=fit_tabular(name,train,val,test,quick=quick)
        elif name in ("lstm","gru","tcn","patchtst"):
            pred,param=fit_deep(name,train,val,test,device,quick=quick)
        elif name=="contrastive_ridge":
            pred,param=fit_contrastive(train,val,test,device,quick=quick)
        else: raise ValueError(f"Unknown model name: {name}")
        predictions[name]=pred;settings["model_parameters"][name]=param
    output.mkdir(parents=True,exist_ok=True)
    prediction_table = pd.DataFrame({"date":test.date,
                                      "previous_observed_price":test.previous_price,
                                      "actual_price":test.price,
                                      "actual_return":actual,
                                      **{f"pred_{k}":v for k,v in predictions.items()}})
    for model, pred in predictions.items():
        prediction_table[f"pred_price_{model}"] = test.previous_price.to_numpy(dtype=float) * np.exp(pred)
    prediction_table.to_csv(output/f"{asset}_test_predictions.csv",index=False)
    rows=[]
    for model,pred in predictions.items():
        actual_price = test.price.to_numpy(dtype=float)
        pred_price = test.previous_price.to_numpy(dtype=float) * np.exp(pred)
        rows.append({"asset":asset,"period":period,"model":model,"n_test":len(test),
                     "test_from":settings["test_from"],"test_to":settings["test_to"],
                     **regression_metrics(actual,pred,train_mean),
                     "price_rmse":float(np.sqrt(np.mean((actual_price-pred_price)**2))),
                     "price_mae":float(np.mean(np.abs(actual_price-pred_price)))})
    (output/f"{asset}_model_config.json").write_text(json.dumps(settings,indent=2))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for asset,filename in (("wti","DCOILWTICO"),("brent","DCOILBRENTEU"),("gas","DHHNGSP")):
        ap.add_argument(f"--{asset}-csv",default=f"data/{filename}.csv")
    ap.add_argument("--abc-csv",default="data/abcnews-date-text.csv")
    ap.add_argument("--period",choices=["abc","full"],default="full",
                    help="abc: match ABC coverage using raw corpus dates; full: latest available FRED history")
    ap.add_argument("--models",default="ridge,elasticnet,histgb,xgboost,lightgbm,lstm,gru,tcn,patchtst,contrastive_ridge")
    ap.add_argument("--output-dir",default="results/model_comparison")
    ap.add_argument("--quick",action="store_true",help="Smoke only: tiny grids/epochs, do not cite as research results")
    args=ap.parse_args()
    seeds()
    try:
        import torch
        torch.set_num_threads(min(4,torch.get_num_threads()))
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device="cpu"
    models=[x.strip() for x in args.models.split(",") if x.strip()]
    filenames={"wti":args.wti_csv,"brent":args.brent_csv,"henry_hub":args.gas_csv}
    output=Path(args.output_dir)
    all_rows=[]
    for asset,filename in filenames.items():
        all_rows.extend(evaluate_one(asset,filename,output,models,args.period,args.abc_csv,device,args.quick))
        summary=pd.DataFrame(all_rows)
        summary.to_csv(output/"model_comparison.csv",index=False)
        print("\n"+summary.to_string(index=False,float_format=lambda x:f"{x:.6f}"),flush=True)
    (output/"benchmark_metadata.json").write_text(json.dumps({
        "seed":SEED,"models":models,"period":args.period,"quick_smoke_test":args.quick,
        "train_fraction":.65,"validation_fraction":.15,"test_fraction":.20,"gap_rows":5,
        "lookback_observed_returns":RETURN_LAGS,"selection":"TimeSeriesSplit on train for tabular; validation for deep/contrastive; test held out",
        "refit":"train plus validation, excluding chronological split gaps","device":str(device),
        "warning":"FRED observation dates are not necessarily real-time publication dates; retrospective research only",
        "version":"v0.5-patchtst"},indent=2))
    print("\nSaved results to",output.resolve())


if __name__=="__main__":
    main()
