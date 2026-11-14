"""Merge separate CPU/neural baseline result CSVs into per-asset RMSE leaderboards."""
from __future__ import annotations
import argparse
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument('csvs',nargs='+')
parser.add_argument('--output',default='results/price_model_leaderboard.csv')
args=parser.parse_args()
all_results=pd.concat([pd.read_csv(p) for p in args.csvs],ignore_index=True)
all_results=all_results.drop_duplicates(['asset','period','model','test_from','test_to'])
for asset,subset in all_results.groupby('asset',sort=False):
    if subset[['test_from','test_to','n_test','period']].drop_duplicates().shape[0] != 1:
        raise ValueError(f'Different test periods for {asset}; do not merge comparisons')
all_results=all_results.sort_values(['asset','rmse','model'])
from pathlib import Path
Path(args.output).parent.mkdir(parents=True,exist_ok=True)
all_results.to_csv(args.output,index=False)
print(all_results[['asset','model','n_test','rmse','mae','oos_r2','directional_accuracy','price_rmse']].to_string(index=False))
print('Saved:',args.output)
