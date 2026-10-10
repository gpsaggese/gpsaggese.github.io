#!/usr/bin/env python
r"""
Run the winner's curse and bootstrap coverage sweeps and save tables and
figures.

The curse sweep runs every (K, N, S) in the grid. The coverage sweep runs
every (N, S) at a fixed K, since coverage is measured per config.

# Usage Example

- Run the default sweep and write results to `skill_luck_sim/results`:
  ```bash
  > run_sweep.py
  ```

- Run a quick smoke version:
  ```bash
  > run_sweep.py --n_reps 50 --n_coverage_reps 10 --n_boot 100 \
      --out_dir tmp.run_sweep
  ```

Import as:

import skill_luck_sim.run_sweep as slsirusw
"""

import argparse
import dataclasses
import itertools
import logging
import os
import time

import numpy as np
import pandas as pd

import skill_luck_sim.bootstrap as slsiboot
import skill_luck_sim.estimate as slsiesti
import skill_luck_sim.generate as slsigene
import skill_luck_sim.plotting as slsiplot

_LOG = logging.getLogger(__name__)

_GRID_K = [3, 5, 10, 20]
_GRID_N = [25, 50, 100, 500]
_GRID_S = [1, 3, 5, 10]


# #############################################################################
# Sweeps
# #############################################################################


def _setting_seed(seed: int, values: list[int]) -> int:
    """
    Derive a seed for one grid setting from the batch seed and the setting.

    This makes each setting's result independent of which other settings
    are run and in which order.
    """
    setting_seed = int(
        np.random.SeedSequence([seed] + values).generate_state(1)[0]
    )
    return setting_seed


def run_curse_sweep(
    base: slsigene.SimParams, n_reps: int, seed: int
) -> pd.DataFrame:
    """
    Run the winner's curse replications over the (K, N, S) grid.

    :param base: parameters other than K, N, S
    :param n_reps: replications per setting
    :param seed: batch seed
    :return: one row per setting with the summary of
        `slsiesti.summarize_replications()`
    """
    rows = []
    for k, n, s in itertools.product(_GRID_K, _GRID_N, _GRID_S):
        params = dataclasses.replace(base, n_configs=k, n_tasks=n, n_seeds=s)
        df = slsiesti.run_replications(
            params, n_reps, _setting_seed(seed, [k, n, s])
        )
        summary = slsiesti.summarize_replications(df, params)
        rows.append({"n_configs": k, "n_tasks": n, "n_seeds": s, **summary})
        _LOG.debug("K=%s N=%s S=%s curse=%.4f", k, n, s, summary["curse"])
    df_out = pd.DataFrame(rows)
    return df_out


def run_coverage_sweep(
    base: slsigene.SimParams, n_reps: int, n_boot: int, seed: int
) -> pd.DataFrame:
    """
    Run bootstrap coverage over the (N, S) grid at K = `base.n_configs`.

    :param base: parameters other than N and S
    :param n_reps: replications per setting
    :param n_boot: bootstrap replicates per interval
    :param seed: batch seed
    :return: one row per (method, N, S)
    """
    dfs = []
    for n, s in itertools.product(_GRID_N, _GRID_S):
        params = dataclasses.replace(base, n_tasks=n, n_seeds=s)
        df = slsiboot.run_coverage(
            params, n_reps, n_boot, _setting_seed(seed, [base.n_configs, n, s])
        )
        df.insert(0, "n_seeds", s)
        df.insert(0, "n_tasks", n)
        dfs.append(df)
    df_out = pd.concat(dfs, ignore_index=True)
    return df_out


# #############################################################################
# Main
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out_dir", action="store", default="skill_luck_sim/results"
    )
    parser.add_argument("--n_reps", action="store", type=int, default=1000)
    parser.add_argument(
        "--n_coverage_reps", action="store", type=int, default=400
    )
    parser.add_argument("--n_boot", action="store", type=int, default=500)
    parser.add_argument("--coverage_k", action="store", type=int, default=5)
    parser.add_argument("--seed", action="store", type=int, default=0)
    parser.add_argument(
        "-v", dest="log_level", action="store", default="INFO", help="Log level"
    )
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    base = slsigene.SimParams()
    _LOG.info("Base parameters: %s", base)
    # Run the curse sweep.
    start = time.time()
    df_curse = run_curse_sweep(base, args.n_reps, args.seed)
    _LOG.info("Curse sweep took %.0f s", time.time() - start)
    df_curse.to_csv(os.path.join(args.out_dir, "curse_sweep.csv"), index=False)
    # Run the coverage sweep.
    start = time.time()
    base_cov = dataclasses.replace(base, n_configs=args.coverage_k)
    df_cov = run_coverage_sweep(
        base_cov, args.n_coverage_reps, args.n_boot, args.seed
    )
    _LOG.info("Coverage sweep took %.0f s", time.time() - start)
    df_cov.to_csv(os.path.join(args.out_dir, "coverage_sweep.csv"), index=False)
    # Plot.
    slsiplot.plot_curse(df_curse, os.path.join(args.out_dir, "fig1.curse.png"))
    slsiplot.plot_prob_best(
        df_curse, os.path.join(args.out_dir, "fig2.prob_best.png")
    )
    slsiplot.plot_coverage(
        df_cov, os.path.join(args.out_dir, "fig3.coverage.png")
    )


if __name__ == "__main__":
    _main(_parse())
