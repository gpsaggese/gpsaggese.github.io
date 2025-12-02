#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PR1 — Electricity Load Forecasting (Full + Per-Model Plots)
-----------------------------------------------------------
Trains: seasonal naive (24), Linear, Ridge, RandomForest, and optional ARIMA, Prophet, LSTM.
- Saves metrics + predictions CSV.
- Saves one PNG per model + a Top-3 overlay + the "best" plot by RMSE.
- CLI toggles: --no-arima --no-prophet --no-lstm
"""
import argparse, sys, math, warnings
from pathlib import Path
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true,float), np.asarray(y_pred,float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100 if mask.sum()>0 else np.nan

def safe_eval_metrics(y_true, y_pred):
    if y_pred is None or len(y_pred)!=len(y_true) or np.isnan(y_pred).any():
        return (np.nan,np.nan,np.nan)
    mae=mean_absolute_error(y_true,y_pred)
    rmse=math.sqrt(mean_squared_error(y_true,y_pred))
    mp=mape(y_true,y_pred)
    return mae,rmse,mp

def save_plot(outdir,name,idx,y_true,y_pred,ylabel):
    plt.figure(figsize=(12,4))
    plt.plot(idx,y_true,label="Actual")
    plt.plot(idx,y_pred,label=f"Predicted ({name})")
    plt.title(f"Actual vs Predicted — {name}")
    plt.xlabel("Time");plt.ylabel(ylabel);plt.legend()
    path=outdir/f"pr1_actual_vs_pred__{name}.png"
    plt.tight_layout();plt.savefig(path,dpi=160);plt.close();return path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--file",required=True);p.add_argument("--timestamp-col",required=True)
    p.add_argument("--target-col",required=True);p.add_argument("--no-arima",action="store_true")
    p.add_argument("--no-prophet",action="store_true");p.add_argument("--no-lstm",action="store_true")
    args=p.parse_args()
    f=Path(args.file);df=pd.read_csv(f)
    df[args.timestamp_col]=pd.to_datetime(df[args.timestamp_col],errors="coerce")
    df=df.dropna(subset=[args.timestamp_col]);df=df.set_index(args.timestamp_col).sort_index()
    df[args.target_col]=pd.to_numeric(df[args.target_col],errors="coerce");df=df.dropna(subset=[args.target_col])
    df=df.resample("H").mean();df[args.target_col]=df[args.target_col].interpolate().bfill()
    def build_features(df,y):
        out=df.copy();out["hour"]=out.index.hour;out["dow"]=out.index.dayofweek
        out["lag24"]=out[y].shift(24);out["roll24"]=out[y].rolling(24,min_periods=12).mean();return out.dropna()
    feat=build_features(df,args.target_col);train,test=feat.iloc[:-168],feat.iloc[-168:]
    Xtr,ytr=train.drop(columns=[args.target_col]).values,train[args.target_col].values
    Xte,yte=test.drop(columns=[args.target_col]).values,test[args.target_col].values
    preds={}
    preds["naive24"]=df[args.target_col].shift(24).reindex(test.index).bfill().values
    preds["linear"]=LinearRegression().fit(Xtr,ytr).predict(Xte)
    preds["ridge"]=Ridge(alpha=1.0).fit(Xtr,ytr).predict(Xte)
    preds["rf"]=RandomForestRegressor(n_estimators=200,n_jobs=-1,random_state=42).fit(Xtr,ytr).predict(Xte)
    if not args.no_arima:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            ar=ARIMA(train[args.target_col],order=(2,1,2)).fit()
            preds["arima"]=ar.forecast(steps=len(test))
        except Exception as e: print("ARIMA failed",e)
    if not args.no_prophet:
        try:
            from prophet import Prophet
            m=Prophet(daily_seasonality=True,weekly_seasonality=True,yearly_seasonality=True)
            d=train.reset_index()[[train.index.name,args.target_col]].rename(columns={train.index.name:"ds",args.target_col:"y"})
            m.fit(d);future=m.make_future_dataframe(periods=len(test),freq="H")
            fc=m.predict(future).set_index("ds")["yhat"].reindex(test.index).interpolate().bfill().ffill()
            preds["prophet"]=fc.values
        except Exception as e: print("Prophet failed",e)
    if not args.no_lstm:
        try:
            import tensorflow as tf;from tensorflow.keras.models import Sequential;from tensorflow.keras.layers import LSTM,Dense
            from sklearn.preprocessing import MinMaxScaler
            sc=MinMaxScaler();yt=sc.fit_transform(ytr.reshape(-1,1))
            Xt=Xtr.reshape((Xtr.shape[0],1,Xtr.shape[1]));Xe=Xte.reshape((Xte.shape[0],1,Xte.shape[1]))
            m=Sequential([LSTM(64,activation="relu",input_shape=(1,Xtr.shape[1])),Dense(1)])
            m.compile(optimizer="adam",loss="mse");m.fit(Xt,yt,epochs=10,batch_size=32,verbose=0)
            yp=sc.inverse_transform(m.predict(Xe,verbose=0)).flatten();preds["lstm"]=yp
        except Exception as e: print("LSTM failed",e)
    metrics={n:safe_eval_metrics(yte,p) for n,p in preds.items()}
    outdir=f.parent;outdir.mkdir(exist_ok=True)
    with open(outdir/"pr1_metrics.txt","w") as fh:
        fh.write("Model\tMAE\tRMSE\tMAPE(%)\n")
        for n,(a,b,c) in metrics.items():
            fh.write(f"{n}\t{a:.4f}\t{b:.4f}\t{c:.2f}\n")
    pd.DataFrame({"y":yte,**{f"yhat_{n}":p for n,p in preds.items()}}).to_csv(outdir/"pr1_predictions.csv")
    for n,p in preds.items():
        if p is not None and len(p)==len(yte) and not np.isnan(p).any():save_plot(outdir,n,test.index,yte,p,args.target_col)
    bests=[(n,v[1]) for n,v in metrics.items() if np.isfinite(v[1])];bests=sorted(bests,key=lambda x:x[1])[:3]
    if bests:
        plt.figure(figsize=(12,4));plt.plot(test.index,yte,label="Actual")
        for n,_ in bests: plt.plot(test.index,preds[n],label=n)
        plt.legend();plt.title("Top3 Models");plt.tight_layout();plt.savefig(outdir/"pr1_actual_vs_pred__top3.png",dpi=160)
        save_plot(outdir,bests[0][0],test.index,yte,preds[bests[0][0]],args.target_col)
if __name__=="__main__":main()
