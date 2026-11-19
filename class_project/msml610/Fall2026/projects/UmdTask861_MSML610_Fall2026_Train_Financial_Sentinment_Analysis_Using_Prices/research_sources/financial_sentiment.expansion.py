# %% [markdown]
# # Additional 2023 development study: A, AMZN and MSFT
# 
# This is a separate coverage-driven study. Two pinned source files failed to supply
# the original 2016–2020 expanded training window. Their failures and the original A
# experiment remain intact. Source boundary dates were inspected, then the common
# 2023-03-09 to 2023-12-01 (exclusive) window and 2023-09-01 cutoff were registered
# before new price queries or outcomes. No company or model was selected by returns.
# 
# The original strict ten-session total-return target is unchanged and still has
# zero eligible labels. This study uses split-adjusted opening-price excess returns,
# excluding cash dividends, with unknown historical headline versions. It is not a
# point-in-time dataset, a new blind final holdout or evidence of trading alpha.
# 
# Run all cells offline using `requirements-course.txt` and the retained private
# archives. The notebook refits the fixed controls in new output directories and
# checks exact reproducibility; it does not collect data, download models, install
# software, read credentials or submit orders.

# %%
import hashlib, json, math, sys, tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, Image
repo=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/"research/financial_sentiment_using_prices/run_historical.py").is_file())
research=repo/"research/financial_sentiment_using_prices";sys.path.insert(0,str(research))
from run_historical import run, NY
from historical_ranges import verify_archive
from price_labels import _connected_groups
root=repo/"reports/financial_sentiment_runs/expanded_ranges_20261021"
source=root/"nasdaq_2023_source";manifest=verify_archive(source)
registration=json.loads((research/"coverage_study_2023_protocol.json").read_text())
assert registration["symbols"]==["A","AMZN","MSFT"]
assert (registration["source_date_start"],registration["source_date_end_exclusive"],registration["test_start"])==("2023-03-09","2023-12-01","2023-09-01")
assert registration["new_price_outcomes_examined_before_registration"] is False
assert manifest["registration"]==registration
output=Path(tempfile.mkdtemp(prefix="notebook_finbert_",dir=repo/"reports/financial_sentiment_runs/finbert_2023_20261021"));(output/".gitignore").write_text("*\n")
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
print("New offline replay:",output.relative_to(repo))

# %% [markdown]
# ## Coverage, chronology and identical controls
# 
# The 9,419 source headlines are vendor-tagged, with no independent relevance gold
# labels. Article bodies were retained only as unavoidable parts of raw CSV ranges;
# they are not features. Actual receipts are from 2026. Local ticker/date ordering
# and both retained boundaries are checked, but global corpus completeness is not
# certified. Empty ticker tags are not company samples.
# 
# Entry is the second following XNYS session after the reported source date; exit
# is ten sessions later. Relative return is company return minus SPY return, with a
# ±0.005 neutral class. Price features use prior-session opening momentum over 5/20
# sessions. Training outcomes must mature before September 1. URL/title proxy groups
# are formed on the full graph before purging. There is no extra fixed embargo;
# within-test overlapping labels are not independent observations.
# 
# Every company uses the same fixed four controls: training mean, price Ridge,
# headline TF-IDF Ridge and price-plus-TF-IDF Ridge. Training-only transforms and
# alpha10/min_df2/max2000/ngrams1,2 remain fixed. Models are evaluated separately per
# company; there is no pooled capital curve or winner-based company selection.

# %%
reports={};splits={};company_rows={};hashes={}
for symbol in registration["symbols"]:
    capture=root/"study_2023"/(symbol+"_capture")
    original=root/"study_2023"/(symbol+"_evaluation")
    target=output/symbol
    reports[symbol]=run(capture,target)
    splits[symbol]=read(target/"split.json");company_rows[symbol]=read(target/"company_days.json")
    hashes[symbol]={name:sha(target/name) for name in ["predictions.json","fitted_models.json","company_days.json","split.json"]}
    assert all(h==sha(original/name) for name,h in hashes[symbol].items())
coverage=pd.DataFrame([{ "company":s,"headlines":r["headlines"],"observed_outcomes":sum(r["label_status_counts"].values()),
    "paired_train":r["train_rows"],"paired_test":r["test_rows"],"maturity_exclusions":r["exclusion_counts"].get("training_outcome_crosses_time_boundary",0),
    "stock_bars":r["price_bars"][s],"strict_eligible":r["strict_prospective_eligible_labels"]} for s,r in reports.items()])
assert coverage["headlines"].sum()==9419
assert coverage["observed_outcomes"].sum()==451
assert coverage["paired_train"].sum()==272 and coverage["paired_test"].sum()==154
assert coverage["maturity_exclusions"].sum()==25 and coverage["strict_eligible"].sum()==0
display(coverage)

# %%
audit=[]
for symbol in registration["symbols"]:
    capture=root/"study_2023"/(symbol+"_capture");m=read(capture/"manifest.json")
    assert m["bars_summary"]["pagination_complete"] is True
    assert sha(capture/"news.json")==manifest["derived_news_sha256"][symbol]
    calendar=read(capture/"calendar.json")["sessions"];prices={}
    for request in m["requests"]:
        for ticker,bars in read(capture/request["file"])["bars"].items():
            for bar in bars:
                day=datetime.fromisoformat(bar["t"].replace("Z","+00:00")).astimezone(NY).date().isoformat()
                assert (ticker,day) not in prices;prices[ticker,day]=bar["o"]
    rows=company_rows[symbol]
    for row in rows:
        i,j=row["entry_session_index"],row["exit_session_index"]
        assert j-i==10
        entry,exit_=calendar[i]["date"],calendar[j]["date"]
        expected=prices[symbol,exit_]/prices[symbol,entry]-prices["SPY",exit_]/prices["SPY",entry]
        assert math.isclose(row["excess_return"],expected,abs_tol=1e-12)
        expected_features=[prices[symbol,calendar[i-1]["date"]]/prices[symbol,calendar[i-1-lag]["date"]]-1 for lag in [5,20]]
        assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(row["price_features"],expected_features))
    split=splits[symbol];groups=_connected_groups(rows)
    shared={groups[r["sample_id"]] for r in split["train"]}&{groups[r["sample_id"]] for r in split["test"]}
    assert not shared
    assert max(r["exit_at"] for r in split["train"])<"2023-09-01T00:00:00Z"
    assert min(r["entry_at"] for r in split["test"])>="2023-09-01T00:00:00Z"
    audit.append({"company":symbol,"first_train_entry":min(r["entry_at"] for r in split["train"]),
        "last_train_outcome":max(r["exit_at"] for r in split["train"]),
        "first_test_entry":min(r["entry_at"] for r in split["test"]),
        "last_test_entry":max(r["entry_at"] for r in split["test"]),"last_test_outcome":max(r["exit_at"] for r in split["test"]),
        "shared_proxy_groups":len(shared)})
display(pd.DataFrame(audit))
print("All 451 labels and prior-price features recomputed from raw observations; no shared retained proxy groups.")

# %% [markdown]
# ## Predictive errors and assumed costs
# 
# Some combined-model errors are smaller than their price-only controls, but gains
# are small or fail to beat training mean/text-only alternatives. No inference of
# significance or alpha follows from this inspected retrospective study. The frozen
# A 2016–2020 FinBERT comparison remains separate; no new FinBERT result is claimed
# for this expanded sample. The transformer has not been fine-tuned.
# 
# Costs are modeled long/cash outcomes on identical non-overlapping intervals within
# each company, not a pooled portfolio. Round-trip 0/10/20bp assumptions are not
# verified broker fees. Dividends, cash interest, shorts and annualization are omitted.

# %%
metric_rows=[{"company":s,"model":name,**m} for s,r in reports.items() for name,m in r["metrics"].items()]
metrics=pd.DataFrame(metric_rows);metrics["mae_pp"]=metrics["mae"]*100;metrics["rmse_pp"]=metrics["rmse"]*100
display(metrics[["company","model","samples","mae_pp","rmse_pp","price_class_balanced_accuracy","price_class_macro_f1"]])
fig,ax=plt.subplots(figsize=(10,5))
metrics.pivot(index="model",columns="company",values="mae_pp").plot.barh(ax=ax)
ax.set_xlabel("Mean absolute excess-price-return error (percentage points; lower is better)")
ax.set_ylabel("");ax.set_title("Separate 2023 retrospective development evaluations")
ax.legend(title="Company");fig.tight_layout();fig.savefig(output/"mae_comparison.png",dpi=180)
display(Image(filename=str(output/"mae_comparison.png")));plt.close(fig)

# %%
cost_rows=[]
for symbol,r in reports.items():
    c=r["cost_evaluation"]
    assert c["actual_broker_charges"] is None and c["cash_dividends_included"] is False
    print(symbol,"non-overlapping intervals",c["eligible_intervals"],"test rows not compounded",c["overlapping_test_rows_not_compounded"])
    for name,scenarios in c["models"].items():
        for scenario in scenarios:cost_rows.append({"company":symbol,"model":name,**scenario})
cost_table=pd.DataFrame(cost_rows);cost_table["price_return_pct"]=cost_table["cumulative_simulated_price_return"]*100
display(cost_table.pivot(index=["company","model"],columns="assumed_roundtrip_cost_bps",values="price_return_pct"))
fig,ax=plt.subplots(figsize=(10,6))
cost_table[cost_table["assumed_roundtrip_cost_bps"]==10].pivot(index="model",columns="company",values="price_return_pct").plot.barh(ax=ax)
ax.set_ylabel("");ax.set_xlabel("Separate-company sparse-interval simulated price return (%)")
ax.set_title("Assumed 10bp round-trip cost; no dividends or pooled capital")
ax.legend(title="Company");fig.tight_layout();fig.savefig(output/"cost_scenarios.png",dpi=180)
display(Image(filename=str(output/"cost_scenarios.png")));plt.close(fig)

# %%
receipt={"status":"Completed offline expanded-study replay","baseline_stage_code_cells":6,"headlines":9419,
    "observed_labels":451,"paired_train":272,"paired_test":154,"maturity_exclusions":25,
    "strict_eligible":0,"alpha_established":False,"core_hashes":hashes,"audit":audit,
    "registration_sha256":sha(research/"coverage_study_2023_protocol.json"),"source_manifest_sha256":sha(source/"manifest.json"),
    "metrics":metric_rows,"output":str(output.relative_to(repo))}
(output/"notebook_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({k:receipt[k] for k in ["status","headlines","observed_labels","paired_train","paired_test","maturity_exclusions","strict_eligible","alpha_established","output"]},indent=2))

# %% [markdown]
# ## Remaining course requirements
# 
# Original total-return/corporate-action and historical-availability evidence remain
# unresolved. The additional registered FinBERT comparison is reproduced below. LM and multiple
# horizons still need their own frozen fair comparison. Transformer fine-tuning, independent attribution gold labels and false
# attribution evaluation remain unfinished. Docker is absent on this Mac; the
# prepared container helper has not built/run. Formal course issue/PR, presentation
# and video still require completion. All work in this study remains local.

# %% [markdown]
# ## Additional registered FinBERT controls: same development rows
# 
# This comparison was registered after the four-control 2023 development outcomes
# had been inspected. It is not a fresh blind test. The fixed packet uses exactly
# the same 426 paired company-days (272 train / 154 test) and distinct original
# headlines per day. No summaries, bodies, prices or outcomes enter text inference.
# 
# The cached quantized model is frozen; mean positive-minus-negative tone is
# calibrated with training-only StandardScaler/Ridge alpha10, separately per company.
# A second model adds the same price features. Historical model availability and
# pretraining overlap are unverified. This is not transformer fine-tuning.
# 
# The bounded CLI used the same cached weights on local CPU, with a 900-second shared
# inference budget and 930-second outer process cap. Any required headline failure
# makes that entire company's added comparator unavailable, retaining all four old
# baselines and denominators. There is no selective row removal. The notebook reuses
# the frozen text receipts and performs no model inference or network access.

# %%
from finbert_comparison import prepare_packet, evaluate_tone_controls, validate_features
import numpy as np
finbert_root=repo/"reports/financial_sentiment_runs/finbert_2023_20261021"
finbert_registration=research/"finbert_2023_protocol.json"
registered=read(finbert_registration)
inference=read(finbert_root/"inference/inference_manifest.json")
assert sha(finbert_registration)==inference["registration_sha256"]
assert registered["new_FinBERT_results_examined_before_registration"] is False
assert inference["rows_selectively_removed"]==0
display(pd.DataFrame(inference["companies"]).T)
unavailable={s:v for s,v in inference["companies"].items() if v["status"]!="Available"}
assert not unavailable, "Required whole-company coverage unavailable; retain baseline denominators and inspect inference_manifest: "+str(unavailable)
comparisons={}
for symbol in registered["companies"]:
    packet_path=finbert_root/"prepare"/symbol/"tone_request.json"
    features_path=finbert_root/"inference"/symbol/"tone_features.json"
    packet,features=read(packet_path),read(features_path)
    assert packet==prepare_packet(root/"study_2023"/(symbol+"_capture"),output/symbol,finbert_registration)
    validate_features(packet,features)
    assert len(features["samples"])==len(splits[symbol]["train"])+len(splits[symbol]["test"])
    report=evaluate_tone_controls(output/symbol,packet_path,features_path,finbert_registration,output/"finbert"/symbol)
    assert read(output/"finbert"/symbol/"predictions.json")==read(finbert_root/"comparison"/symbol/"predictions.json")
    assert (output/"finbert"/symbol/"fitted_tone_models.json").read_bytes()==(finbert_root/"comparison"/symbol/"fitted_tone_models.json").read_bytes()
    tone={r["sample_id"]:r["tone_score"] for r in features["samples"]}
    fitted=read(output/"finbert"/symbol/"fitted_tone_models.json")
    for name,price in [("finbert_tone_ridge",False),("price_finbert_ridge",True)]:
        design=[(r["price_features"] if price else [])+[tone[r["sample_id"]]] for r in splits[symbol]["train"]]
        assert np.allclose(fitted[name]["means"],np.mean(design,axis=0),rtol=0,atol=1e-12)
        assert fitted[name]["training_sample_ids"]==[r["sample_id"] for r in splits[symbol]["train"]]
    assert report["cost_evaluation"]["selected_sample_ids"]==reports[symbol]["cost_evaluation"]["selected_sample_ids"]
    comparisons[symbol]=report
print("Complete paired rows:",sum(r["train_rows"]+r["test_rows"] for r in comparisons.values()))
print("Actual unique offline inferences:",inference["actual_unique_inferences"],"; elapsed seconds:",inference["elapsed_wall_seconds"])


# %%
six_metrics=pd.DataFrame([{"company":s,"model":name,**m} for s,r in comparisons.items() for name,m in r["metrics"].items()])
six_metrics["mae_pp"]=100*six_metrics["mae"];six_metrics["rmse_pp"]=100*six_metrics["rmse"]
display(six_metrics[["company","model","samples","mae_pp","rmse_pp","price_class_balanced_accuracy","price_class_macro_f1"]])
fig,ax=plt.subplots(figsize=(11,6))
six_metrics.pivot(index="model",columns="company",values="mae_pp").plot.barh(ax=ax)
ax.set_ylabel("");ax.set_xlabel("MAE in excess-price-return percentage points; lower is better")
ax.set_title("Six fixed controls on exactly the same inspected development rows")
ax.legend(title="Company",loc="upper left",bbox_to_anchor=(1,1));fig.tight_layout()
fig.savefig(output/"finbert_mae_comparison.png",dpi=180)
display(Image(filename=str(output/"finbert_mae_comparison.png")));plt.close(fig)
cost_rows=[]
for symbol,r in comparisons.items():
    c=r["cost_evaluation"]
    for name in ["finbert_tone_ridge","price_finbert_ridge","price_ridge","price_text_ridge","always_company","always_spy"]:
        for scenario in c["models"][name]:cost_rows.append({"company":symbol,"model":name,**scenario})
costs=pd.DataFrame(cost_rows);costs["simulated_price_return_pct"]=100*costs["cumulative_simulated_price_return"]
display(costs.pivot(index=["company","model"],columns="assumed_roundtrip_cost_bps",values="simulated_price_return_pct"))
for symbol in ["AMZN","MSFT"]:
    c=comparisons[symbol]["cost_evaluation"]["models"]
    assert c["finbert_tone_ridge"]==c["always_company"] and c["price_finbert_ridge"]==c["always_company"]
print("Both new controls equal always-company exposure on AMZN/MSFT cost intervals; A new controls take no company positions.")


# %%
final_receipt={"status":"Registered 2023 FinBERT comparison reproduced offline","notebook_code_cells":9,
    "paired_train":272,"paired_test":154,"new_rows_removed":0,"coverage_failures":unavailable,
    "registration_sha256":sha(finbert_registration),"inference_manifest_sha256":sha(finbert_root/"inference/inference_manifest.json"),
    "prediction_hashes":{s:sha(output/"finbert"/s/"predictions.json") for s in comparisons},
    "six_model_metrics":six_metrics.to_dict(orient="records"),"strict_eligible":0,"alpha_established":False,
    "comparison_scope":"Added comparator after old development outcomes inspected; not a fresh blind test"}
(output/"finbert_comparison_receipt.json").write_text(json.dumps(final_receipt,indent=2)+"\n")
print(json.dumps({k:final_receipt[k] for k in ["status","notebook_code_cells","paired_train","paired_test","new_rows_removed","coverage_failures","alpha_established"]},indent=2))


# %% [markdown]
# ## Interpretation and exact remaining course scope
# 
# Adding FinBERT to price features worsens A and AMZN MAE. MSFT has a small reduction;
# no significance, fresh holdout or alpha claim follows. Both new controls take no
# company positions on A's selected cost intervals and equal always-company on all
# AMZN/MSFT cost intervals. Positive simulated returns therefore do not demonstrate
# incremental text timing. Assumed fees remain separate from unknown actual charges.
# 
# The official research idea's implementation plan asks for multiple horizons with
# raw and excess-return labels, and a predictive-power table across horizons. It does
# not mandate exact 5/10/20-session choices. Attribution requires large-move detection,
# preceding-news ranking, an unexplained output, validation against hand-labeled known
# event days, and unexplained/false-attribution rates. The document does not specify
# annotator count or blinding; independent, model-blinded annotation is the project's
# evaluation requirement, not an invented quotation. These experiments were not run
# in this bounded FinBERT slice.
# 
# [Official research idea, implementation plan](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/research/ideas/draft.Train_Financial_Sentinment_Analysis_Using_Prices.md#implementation-plan).
# The source is a draft project idea; it is distinct from the general grading rubric.
# LM, horizons, transformer fine-tuning, independent attribution, original strict
# evidence, Docker execution and final course submission/video remain unfinished.
