"""
Unit tests for `causal_success_analysis_utils.py`.

The fast tests use small runs or synthetic data. The slow tests freeze the
results of the baseline experiment shown in `causal_success_example.ipynb`
(100 agents, 80 periods, seed 42). None of the tests needs PyMC.

Import as:

import research.Causal_Analysis_of_Agent_Skill_And_Luck.test.test_causal_success_analysis_utils as rcaoasalttcsau
"""

import logging
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import helpers.hunit_test as hunitest

# Draw figures without a display.
matplotlib.use("Agg")

import research.Causal_Analysis_of_Agent_Skill_And_Luck.causal_success_analysis_utils as rcaoasalcsau
import research.Causal_Analysis_of_Agent_Skill_And_Luck.causal_success_utils as rcaoasalcsu

_LOG = logging.getLogger(__name__)


def _df_to_str(df: pd.DataFrame) -> str:
    """
    Represent a table as CSV with 4 decimals, to compare it as a whole.

    :param df: table to represent
    :return: CSV text with `\\n` line endings on every platform
    """
    txt = df.round(4).to_csv(lineterminator="\n")
    return txt


def _get_figure_titles() -> str:
    """
    Get the title and the panel titles of the current figure, then close it.

    :return: one title per line, the figure title first
    """
    fig = plt.gcf()
    titles = [fig.get_suptitle()] + [ax.get_title() for ax in fig.axes]
    plt.close(fig)
    txt = "\n".join(titles)
    return txt


def _get_small_config() -> Dict[str, Any]:
    """
    Get the default config with 20 agents and 10 periods, for fast tests.

    :return: config for `run_experiment()`
    """
    config = rcaoasalcsu.get_default_config()
    config["population"]["n_agents"] = 20
    config["events"]["n_periods"] = 10
    return config


def _get_start_below_floor_config() -> Dict[str, Any]:
    """
    Get a config of 10 agents whose `initial_capital` is below the floor, and
    no events.

    The simulator raises the `initial_capital` of 1 to the floor of 2, so the
    agents start with 2 and keep it.

    :return: config for `run_experiment()`
    """
    config = rcaoasalcsu.get_default_config()
    config["population"]["n_agents"] = 10
    config["population"]["initial_capital"] = 1.0
    config["events"]["capital_floor"] = 2.0
    config["events"]["n_lucky_per_period"] = 0
    config["events"]["n_unlucky_per_period"] = 0
    return config


def _make_synthetic_causal_df(true_effect: float, seed: int) -> pd.DataFrame:
    """
    Build a table where log-capital depends on the treatment with a known
    constant effect.

    The treatment `lucky_events` depends on the talents, which also change
    log-capital, so a naive regression is confounded.

    :param true_effect: effect of one unit of `lucky_events` on log-capital
    :param seed: seed of the random draws
    :return: table with the columns of `get_results_dataframe()` used by the
        causal estimators
    """
    rng = np.random.default_rng(seed)
    n_rows = 300
    talents = rng.uniform(0.0, 1.0, size=(n_rows, 3))
    lucky_events = 3.0 * talents[:, 0] + rng.normal(0.0, 1.0, size=n_rows)
    log_capital = (
        true_effect * lucky_events
        + np.sin(3.0 * talents[:, 0])
        + talents[:, 1]
        + rng.normal(0.0, 0.1, size=n_rows)
    )
    df = pd.DataFrame(
        {
            "id": np.arange(n_rows),
            "talent_intensity": talents[:, 0],
            "talent_iq": talents[:, 1],
            "talent_networking": talents[:, 2],
            "lucky_events": lucky_events,
            "capital": np.exp(log_capital),
        }
    )
    return df


# #############################################################################
# Test_cell1_1_config_to_df
# #############################################################################


class Test_cell1_1_config_to_df(hunitest.TestCase):
    """
    Test the `cell1_1_config_to_df()` function.
    """

    def test1(self) -> None:
        """
        Test that a top-level value and a group become one row each.
        """
        # Prepare inputs.
        config = {
            "seed": 3,
            "events": {"n_periods": 5, "lucky_clip": (0.1, 0.2)},
        }
        # Prepare outputs.
        expected = """
        ,group,parameter,value
        0,experiment,seed,3
        1,events,n_periods,5
        2,events,lucky_clip,"(0.1, 0.2)"
        """
        # Run test.
        df = rcaoasalcsau.cell1_1_config_to_df(config)
        # Check outputs.
        actual = _df_to_str(df)
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_cell1_1_get_setup_txt
# #############################################################################


class Test_cell1_1_get_setup_txt(hunitest.TestCase):
    """
    Test the `cell1_1_get_setup_txt()` function.
    """

    def test1(self) -> None:
        """
        Test the set-up of the default config.
        """
        # Prepare inputs.
        config = rcaoasalcsu.get_default_config()
        # Prepare outputs.
        expected = "100 agents, 80 periods, seed 42"
        # Run test.
        actual = rcaoasalcsau.cell1_1_get_setup_txt(config)
        # Check outputs.
        self.assert_equal(actual, expected)


# #############################################################################
# Test_cell1_3_plot_talent_distributions
# #############################################################################


class Test_cell1_3_plot_talent_distributions(hunitest.TestCase):
    """
    Test the `cell1_3_plot_talent_distributions()` function.
    """

    def test1(self) -> None:
        """
        Test that each talent has a panel titled with its std.
        """
        # Prepare inputs.
        config = _get_small_config()
        df = rcaoasalcsu.run_experiment(config)["df"]
        setup_txt = "setup"
        # Prepare outputs.
        expected = """
        Talents of the agents (setup)
        Intensity (std 0.123)
        IQ (std 0.132)
        Networking (std 0.098)
        """
        # Run test.
        rcaoasalcsau.cell1_3_plot_talent_distributions(df, setup_txt)
        # Check outputs.
        actual = _get_figure_titles()
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_cell2_1_plot_capital_distribution
# #############################################################################


class Test_cell2_1_plot_capital_distribution(hunitest.TestCase):
    """
    Test the `cell2_1_plot_capital_distribution()` function.
    """

    def test1(self) -> None:
        """
        Test that the panels are titled with the Gini, the skewness, and the
        shares.
        """
        # Prepare inputs.
        config = _get_small_config()
        results = rcaoasalcsu.run_experiment(config)
        setup_txt = "setup"
        # Prepare outputs.
        expected = """
        Final capital (setup)
        Capital (Gini 0.176, skewness 0.87)
        log(capital) (skewness 0.14)
        Lorenz (top 10% 17.2%, bottom 50% 37.9%)
        """
        # Run test.
        rcaoasalcsau.cell2_1_plot_capital_distribution(
            results["df"], results["stats"], setup_txt
        )
        # Check outputs.
        actual = _get_figure_titles()
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_cell3_2_compute_log_capital_correlations
# #############################################################################


class Test_cell3_2_compute_log_capital_correlations(hunitest.TestCase):
    """
    Test the `cell3_2_compute_log_capital_correlations()` function.
    """

    def test1(self) -> None:
        """
        Test the correlations when log-capital is proportional to the net
        events.
        """
        # Prepare inputs.
        df = pd.DataFrame(
            {
                "talent_norm": [1.1, 1.3, 1.2, 1.5, 1.4, 1.0],
                "talent_intensity": [0.4, 0.6, 0.5, 0.7, 0.3, 0.2],
                "talent_iq": [0.5, 0.4, 0.6, 0.8, 0.7, 0.3],
                "talent_networking": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                "lucky_events": [1, 3, 2, 5, 4, 0],
                "unlucky_events": [2, 1, 3, 1, 2, 4],
            }
        )
        df["net_events"] = df["lucky_events"] - df["unlucky_events"]
        df["capital"] = np.exp(0.2 * df["net_events"])
        # Prepare outputs.
        # The net events explain log-capital exactly, so their `r` is 1.
        expected = """
        predictor,kind,r,ci_lower,ci_upper,p_value
        talent_norm,talent,0.9667,0.7202,0.9965,0.0016
        talent_intensity,talent,0.7436,-0.1713,0.9699,0.0902
        talent_iq,talent,0.7436,-0.1713,0.9699,0.0902
        talent_networking,talent,0.1859,-0.7368,0.8667,0.7243
        lucky_events,events,0.9667,0.7202,0.9965,0.0016
        unlucky_events,events,-0.9124,-0.9905,-0.3884,0.0112
        net_events,events,1.0,1.0,1.0,0.0
        """
        # Run test.
        corr_df = rcaoasalcsau.cell3_2_compute_log_capital_correlations(df)
        # Check outputs.
        actual = _df_to_str(corr_df)
        self.assert_equal(actual, expected, dedent=True)

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Freeze the correlations of the baseline experiment.
        """
        # Prepare inputs.
        config = rcaoasalcsu.get_default_config()
        df = rcaoasalcsu.run_experiment(config)["df"]
        # Prepare outputs.
        expected = """
        predictor,kind,r,ci_lower,ci_upper,p_value
        talent_norm,talent,-0.0447,-0.239,0.153,0.6587
        talent_intensity,talent,-0.0417,-0.2361,0.156,0.6807
        talent_iq,talent,0.0856,-0.1127,0.2774,0.3971
        talent_networking,talent,-0.1028,-0.2933,0.0956,0.309
        lucky_events,events,0.5397,0.384,0.6656,0.0
        unlucky_events,events,-0.5962,-0.7095,-0.4528,0.0
        net_events,events,0.9199,0.883,0.9455,0.0
        """
        # Run test.
        corr_df = rcaoasalcsau.cell3_2_compute_log_capital_correlations(df)
        # Check outputs.
        actual = _df_to_str(corr_df)
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_cell3_3_plot_talent_and_luck_vs_log_capital
# #############################################################################


class Test_cell3_3_plot_talent_and_luck_vs_log_capital(hunitest.TestCase):
    """
    Test the `cell3_3_plot_talent_and_luck_vs_log_capital()` function.
    """

    def test1(self) -> None:
        """
        Test that the panels are titled with the correlations and their
        intervals.
        """
        # Prepare inputs.
        config = _get_small_config()
        df = rcaoasalcsu.run_experiment(config)["df"]
        corr_df = rcaoasalcsau.cell3_2_compute_log_capital_correlations(df)
        setup_txt = "setup"
        # Prepare outputs.
        expected = """
        Talent and luck against log-capital (setup)
        Talent norm
        r = 0.01, 95% CI [-0.43, 0.45]
        Realized beneficial events
        r = 0.70, 95% CI [0.37, 0.87]
        """
        # Run test.
        rcaoasalcsau.cell3_3_plot_talent_and_luck_vs_log_capital(
            df, corr_df, setup_txt
        )
        # Check outputs.
        actual = _get_figure_titles()
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_cell3_4_add_ranks
# #############################################################################


class Test_cell3_4_add_ranks(hunitest.TestCase):
    """
    Test the `cell3_4_add_ranks()` function.
    """

    def test1(self) -> None:
        """
        Test that rank 1 is the most talented and the richest agent.
        """
        # Prepare inputs.
        df = pd.DataFrame(
            {
                "id": [0, 1, 2],
                "talent_norm": [1.4, 1.2, 1.3],
                "capital": [0.5, 2.0, 1.0],
            }
        )
        # Prepare outputs.
        expected = """
        ,id,talent_norm,capital,talent_rank,capital_rank
        0,0,1.4,0.5,1.0,3.0
        1,1,1.2,2.0,3.0,1.0
        2,2,1.3,1.0,2.0,2.0
        """
        # Run test.
        df_ranked = rcaoasalcsau.cell3_4_add_ranks(df)
        # Check outputs.
        actual = _df_to_str(df_ranked)
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_cell3_5_summarize_top_performers
# #############################################################################


class Test_cell3_5_summarize_top_performers(hunitest.TestCase):
    """
    Test the `cell3_5_summarize_top_performers()` function.
    """

    def test1(self) -> None:
        """
        Test the summary when the richest agent is the least talented but one.
        """
        # Prepare inputs.
        df = pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "talent_norm": [1.4, 1.2, 1.3, 1.1],
                "capital": [0.5, 2.0, 1.0, 0.8],
                "lucky_events": [1, 4, 2, 2],
                "unlucky_events": [3, 1, 2, 2],
            }
        )
        df["net_events"] = df["lucky_events"] - df["unlucky_events"]
        df_ranked = rcaoasalcsau.cell3_4_add_ranks(df)
        n_top = 1
        # Prepare outputs.
        expected = """
        group,n_agents,mean_talent_rank,median_talent_rank,mean_capital,mean_lucky_events,mean_unlucky_events,mean_net_events
        top,1,3.0,3.0,2.0,4.0,1.0,3.0
        others,3,2.3333,2.0,0.7667,1.6667,2.3333,-0.6667
        all,4,2.5,2.5,1.075,2.25,2.0,0.25
        """
        # Run test.
        summary_df = rcaoasalcsau.cell3_5_summarize_top_performers(
            df_ranked, n_top
        )
        # Check outputs.
        actual = _df_to_str(summary_df)
        self.assert_equal(actual, expected, dedent=True)

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Freeze the top 10 performers of the baseline experiment.
        """
        # Prepare inputs.
        config = rcaoasalcsu.get_default_config()
        df = rcaoasalcsu.run_experiment(config)["df"]
        df_ranked = rcaoasalcsau.cell3_4_add_ranks(df)
        n_top = 10
        # Prepare outputs.
        expected = """
        group,n_agents,mean_talent_rank,median_talent_rank,mean_capital,mean_lucky_events,mean_unlucky_events,mean_net_events
        top,10,42.6,36.0,1.6358,4.0,1.8,2.2
        others,90,51.3778,52.5,0.8054,2.1333,4.2444,-2.1111
        all,100,50.5,50.5,0.8884,2.32,4.0,-1.68
        """
        # Run test.
        summary_df = rcaoasalcsau.cell3_5_summarize_top_performers(
            df_ranked, n_top
        )
        # Check outputs.
        actual = _df_to_str(summary_df)
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_cell4_2_compute_log_change_per_event
# #############################################################################


class Test_cell4_2_compute_log_change_per_event(hunitest.TestCase):
    """
    Test the `cell4_2_compute_log_change_per_event()` function.
    """

    def helper(
        self,
        events: List[Tuple[str, float]],
        capital_floor: float,
        expected: str,
    ) -> None:
        """
        Apply events by hand to one agent and check the log-change table.

        :param events: `(event_type, impact)` pairs applied in order
        :param capital_floor: capital floor of the agent
        :param expected: expected table, as CSV
        """
        # Prepare inputs.
        intensity = 0.5
        iq = 0.5
        networking = 0.5
        agent = rcaoasalcsu.Agent(
            0, intensity, iq, networking, capital_floor=capital_floor
        )
        for event_type, impact in events:
            agent.apply_event(event_type, impact)
        agents = [agent]
        # Run test.
        log_change_df = rcaoasalcsau.cell4_2_compute_log_change_per_event(agents)
        # Check outputs.
        actual = _df_to_str(log_change_df)
        self.assert_equal(actual, expected, dedent=True)

    def test1(self) -> None:
        """
        Test the log changes of lucky and unlucky events applied by hand.
        """
        # Prepare inputs.
        events = [("lucky", 0.25), ("unlucky", 0.1), ("lucky", 0.5)]
        capital_floor = 0.01
        # Prepare outputs.
        # The gains are log(1.25) and log(1.5), the loss is log(0.9).
        expected = """
        event_kind,n_events,mean_log_change,std_log_change,min_log_change,max_log_change
        beneficial,2,0.3143,0.0912,0.2231,0.4055
        detrimental,1,-0.1054,0.0,-0.1054,-0.1054
        unchanged,0,,,,
        """
        # Run test.
        self.helper(events, capital_floor, expected)

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Freeze the log changes of the events of the baseline experiment.
        """
        # Prepare inputs.
        config = rcaoasalcsu.get_default_config()
        agents = rcaoasalcsu.run_experiment(config)["agents"]
        # Prepare outputs.
        expected = """
        event_kind,n_events,mean_log_change,std_log_change,min_log_change,max_log_change
        beneficial,232,0.1994,0.0724,0.0343,0.3733
        detrimental,400,-0.1644,0.0563,-0.3161,-0.0513
        unchanged,0,,,,
        """
        # Run test.
        log_change_df = rcaoasalcsau.cell4_2_compute_log_change_per_event(agents)
        # Check outputs.
        actual = _df_to_str(log_change_df)
        self.assert_equal(actual, expected, dedent=True)

    def test3(self) -> None:
        """
        Test a run with no beneficial event: its statistics are undefined.
        """
        # Prepare inputs.
        events = [("unlucky", 0.1), ("unlucky", 0.2)]
        capital_floor = 0.01
        # Prepare outputs.
        expected = """
        event_kind,n_events,mean_log_change,std_log_change,min_log_change,max_log_change
        beneficial,0,,,,
        detrimental,2,-0.1643,0.0589,-0.2231,-0.1054
        unchanged,0,,,,
        """
        # Run test.
        self.helper(events, capital_floor, expected)

    def test4(self) -> None:
        """
        Test an unlucky event on an agent already at the floor: the capital
        does not change.
        """
        # Prepare inputs.
        # The first event takes the capital from 1.0 to the floor of 0.5,
        # instead of 0.4, and the second one leaves it at 0.5.
        events = [("unlucky", 0.6), ("unlucky", 0.2)]
        capital_floor = 0.5
        # Prepare outputs.
        expected = """
        event_kind,n_events,mean_log_change,std_log_change,min_log_change,max_log_change
        beneficial,0,,,,
        detrimental,1,-0.6931,0.0,-0.6931,-0.6931
        unchanged,1,0.0,0.0,0.0,0.0
        """
        # Run test.
        self.helper(events, capital_floor, expected)


# #############################################################################
# Test_cell4_3_estimate_dml_effect
# #############################################################################


class Test_cell4_3_estimate_dml_effect(hunitest.TestCase):
    """
    Test the `cell4_3_estimate_dml_effect()` function.
    """

    def test1(self) -> None:
        """
        Test that the 95% interval contains a known effect under confounding.
        """
        # Prepare inputs.
        true_effect = 0.2
        seed = 0
        df = _make_synthetic_causal_df(true_effect, seed)
        # Run test.
        dml_df = rcaoasalcsau.cell4_3_estimate_dml_effect(df, seed)
        # Check outputs.
        row = dml_df.iloc[0]
        self.assertLess(row["ci_lower"], true_effect)
        self.assertLess(true_effect, row["ci_upper"])

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Freeze the DML effect of the baseline experiment.
        """
        # Prepare inputs.
        config = rcaoasalcsu.get_default_config()
        df = rcaoasalcsu.run_experiment(config)["df"]
        seed = config["seed"]
        # Prepare outputs.
        expected = """
        estimator,effect,stderr,ci_lower,ci_upper,p_value,pct_change
        LinearDML,0.1567,0.026,0.1058,0.2076,0.0,0.1697
        """
        # Run test.
        dml_df = rcaoasalcsau.cell4_3_estimate_dml_effect(df, seed)
        # Check outputs.
        actual = _df_to_str(dml_df)
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_cell4_4_estimate_cate
# #############################################################################


class Test_cell4_4_estimate_cate(hunitest.TestCase):
    """
    Test the `cell4_4_estimate_cate()` function.
    """

    def test1(self) -> None:
        """
        Test that the interval of the average effect contains a known constant
        effect.
        """
        # Prepare inputs.
        true_effect = 0.2
        seed = 0
        df = _make_synthetic_causal_df(true_effect, seed)
        # Prepare outputs.
        expected_columns = (
            "['id', 'talent_intensity', 'talent_iq', 'talent_networking', "
            "'cate', 'ci_lower', 'ci_upper']"
        )
        # Run test.
        cate_df, summary_df = rcaoasalcsau.cell4_4_estimate_cate(df, seed)
        # Check outputs.
        self.assert_equal(str(cate_df.columns.tolist()), expected_columns)
        self.assertEqual(len(cate_df), len(df))
        row = summary_df.iloc[0]
        self.assertLess(row["ci_lower_mean"], true_effect)
        self.assertLess(true_effect, row["ci_upper_mean"])

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Freeze the causal forest summary of the baseline experiment.
        """
        # Prepare inputs.
        config = rcaoasalcsu.get_default_config()
        df = rcaoasalcsu.run_experiment(config)["df"]
        seed = config["seed"]
        # Prepare outputs.
        expected = """
        estimator,mean_cate,stderr_mean,ci_lower_mean,ci_upper_mean,std_cate,mean_ci_width,share_ci_excludes_mean
        CausalForestDML,0.1594,0.0331,0.0945,0.2242,0.0134,0.1261,0.0
        """
        # Run test.
        _, summary_df = rcaoasalcsau.cell4_4_estimate_cate(df, seed)
        # Check outputs.
        actual = _df_to_str(summary_df)
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_cell4_5_plot_cate
# #############################################################################


class Test_cell4_5_plot_cate(hunitest.TestCase):
    """
    Test the `cell4_5_plot_cate()` function.
    """

    def test1(self) -> None:
        """
        Test that the figure has the sorted panel and the IQ panel.
        """
        # Prepare inputs.
        cate_df = pd.DataFrame(
            {
                "id": [0, 1, 2],
                "talent_intensity": [0.4, 0.5, 0.6],
                "talent_iq": [0.3, 0.5, 0.7],
                "talent_networking": [0.5, 0.5, 0.5],
                "cate": [0.1, 0.2, 0.15],
                "ci_lower": [0.0, 0.1, 0.05],
                "ci_upper": [0.2, 0.3, 0.25],
            }
        )
        dml_df = pd.DataFrame(
            {
                "estimator": ["LinearDML"],
                "effect": [0.15],
                "ci_lower": [0.11],
                "ci_upper": [0.19],
            }
        ).set_index("estimator")
        setup_txt = "setup"
        # Prepare outputs.
        expected = """
        Effect of one realized beneficial event (setup)
        Per-agent effects with their intervals
        Per-agent effects against IQ
        """
        # Run test.
        rcaoasalcsau.cell4_5_plot_cate(cate_df, dml_df, setup_txt)
        # Check outputs.
        actual = _get_figure_titles()
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_cell5_2_run_policy_comparison
# #############################################################################


class Test_cell5_2_run_policy_comparison(hunitest.TestCase):
    """
    Test the `cell5_2_run_policy_comparison()` function.
    """

    def test1(self) -> None:
        """
        Test that the performance policy equals the egalitarian one when every
        agent starts with the same capital.
        """
        # Prepare inputs.
        config = _get_small_config()
        cate_values = np.ones(config["population"]["n_agents"])
        # Prepare outputs.
        expected_policies = (
            "['none', 'egalitarian', 'meritocratic', 'performance', "
            "'random', 'cate_optimal']"
        )
        # Run test.
        policy_df, policy_results = rcaoasalcsau.cell5_2_run_policy_comparison(
            config, cate_values
        )
        # Check outputs.
        self.assert_equal(str(policy_df.index.tolist()), expected_policies)
        self.assert_equal(
            str(sorted(policy_results)), str(sorted(policy_df.index))
        )
        self.assert_equal(
            policy_df.loc["performance"].to_string(),
            policy_df.loc["egalitarian"].to_string(),
        )

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Freeze the policy comparison of the baseline experiment, with the CATE
        of the causal forest.
        """
        # Prepare inputs.
        config = rcaoasalcsu.get_default_config()
        df = rcaoasalcsu.run_experiment(config)["df"]
        cate_df, _ = rcaoasalcsau.cell4_4_estimate_cate(df, config["seed"])
        cate_values = cate_df["cate"].to_numpy()
        # Prepare outputs.
        expected = """
        policy,total_initial_capital,total_final_capital,final_to_initial_ratio,mean_capital,median_capital,gini,top_10_pct_share,bottom_50_pct_share
        none,100.0,88.8435,0.8884,0.8884,0.8161,0.2171,0.1841,0.345
        egalitarian,200.0,177.6869,0.8884,1.7769,1.6321,0.2171,0.1841,0.345
        meritocratic,200.0,177.6237,0.8881,1.7762,1.6472,0.2173,0.1853,0.3453
        performance,200.0,177.6869,0.8884,1.7769,1.6321,0.2171,0.1841,0.345
        random,200.0,229.3635,1.1468,2.2936,0.8161,0.6904,0.684,0.1336
        cate_optimal,200.0,177.9731,0.8899,1.7797,1.6901,0.2206,0.1858,0.3436
        """
        # Run test.
        policy_df, _ = rcaoasalcsau.cell5_2_run_policy_comparison(
            config, cate_values
        )
        # Check outputs.
        actual = _df_to_str(policy_df)
        self.assert_equal(actual, expected, dedent=True)

    def test3(self) -> None:
        """
        Test that the total initial capital is the actual one when
        `initial_capital` is below the floor.

        The config gives 10 * 1 + 100 = 110, but the agents start with 2, so
        the totals are 20 and 120.
        """
        # Prepare inputs.
        config = _get_start_below_floor_config()
        cate_values = np.ones(config["population"]["n_agents"])
        # Prepare outputs.
        # With no events, every final capital is the starting capital.
        expected = """
        policy,total_initial_capital,total_final_capital,final_to_initial_ratio
        none,20.0,20.0,1.0
        egalitarian,120.0,120.0,1.0
        meritocratic,120.0,120.0,1.0
        performance,120.0,120.0,1.0
        random,120.0,120.0,1.0
        cate_optimal,120.0,120.0,1.0
        """
        # Run test.
        policy_df, _ = rcaoasalcsau.cell5_2_run_policy_comparison(
            config, cate_values
        )
        # Check outputs.
        columns = [
            "total_initial_capital",
            "total_final_capital",
            "final_to_initial_ratio",
        ]
        actual = _df_to_str(policy_df[columns])
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_cell5_3_compute_implied_initial_capital
# #############################################################################


class Test_cell5_3_compute_implied_initial_capital(hunitest.TestCase):
    """
    Test the `cell5_3_compute_implied_initial_capital()` function.
    """

    def helper(
        self, config: Dict[str, Any], cate_values: np.ndarray, expected: str
    ) -> None:
        """
        Run the policies of `config` and check the implied initial capital.

        :param config: baseline config, with no policy
        :param cate_values: per-agent effects of the `cate_optimal` policy
        :param expected: expected table, as CSV
        """
        # Prepare inputs.
        _, policy_results = rcaoasalcsau.cell5_2_run_policy_comparison(
            config, cate_values
        )
        # Run test.
        implied_df = rcaoasalcsau.cell5_3_compute_implied_initial_capital(
            policy_results
        )
        # Check outputs.
        actual = _df_to_str(implied_df)
        self.assert_equal(actual, expected, dedent=True)

    def test1(self) -> None:
        """
        Test that, with no agent at the floor, every policy keeps the events and
        only changes the starting capital.
        """
        # Prepare inputs.
        config = _get_small_config()
        cate_values = np.linspace(0.0, 1.0, config["population"]["n_agents"])
        # Prepare outputs.
        # 20 agents start with 1.0 each and the policies add a budget of 100.
        expected = """
        policy,same_event_counts,n_agents_at_floor,rescaling_holds,sum_actual_initial,min_implied_initial,mean_implied_initial,max_implied_initial,sum_implied_initial
        none,True,0,True,20.0,1.0,1.0,1.0,20.0
        egalitarian,True,0,True,120.0,6.0,6.0,6.0,120.0
        meritocratic,True,0,True,120.0,5.5664,6.0,6.3483,120.0
        performance,True,0,True,120.0,6.0,6.0,6.0,120.0
        random,True,0,True,120.0,1.0,6.0,101.0,120.0
        cate_optimal,True,0,True,120.0,1.0,6.0,11.0,120.0
        """
        # Run test.
        self.helper(config, cate_values, expected)

    def test2(self) -> None:
        """
        Test that a binding capital floor disables the inference.

        Without the guard, this config inferred a total starting capital of
        53.1 for `egalitarian`, while the agents started with 110.
        """
        # Prepare inputs.
        config = rcaoasalcsu.get_default_config()
        config["population"]["n_agents"] = 10
        config["events"]["capital_floor"] = 0.5
        cate_values = np.ones(config["population"]["n_agents"])
        # Prepare outputs.
        # The events are the same, but the floor binds in every run.
        expected = """
        policy,same_event_counts,n_agents_at_floor,rescaling_holds,sum_actual_initial,min_implied_initial,mean_implied_initial,max_implied_initial,sum_implied_initial
        none,True,8,False,10.0,,,,
        egalitarian,True,3,False,110.0,,,,
        meritocratic,True,3,False,110.0,,,,
        performance,True,3,False,110.0,,,,
        random,True,7,False,110.0,,,,
        cate_optimal,True,3,False,110.0,,,,
        """
        # Run test.
        self.helper(config, cate_values, expected)

    def test3(self) -> None:
        """
        Test that the inference uses the actual starting capital when
        `initial_capital` is below the floor.

        With the configured `initial_capital` of 1, the helper inferred a total
        starting capital of 60 for every policy, while the agents started with
        120.
        """
        # Prepare inputs.
        config = _get_start_below_floor_config()
        cate_values = np.ones(config["population"]["n_agents"])
        # Prepare outputs.
        # Starting at the floor is not an event, so the rescaling holds and the
        # implied capital matches the actual one: 2 per agent, plus the
        # allocation.
        expected = """
        policy,same_event_counts,n_agents_at_floor,rescaling_holds,sum_actual_initial,min_implied_initial,mean_implied_initial,max_implied_initial,sum_implied_initial
        none,True,0,True,20.0,2.0,2.0,2.0,20.0
        egalitarian,True,0,True,120.0,12.0,12.0,12.0,120.0
        meritocratic,True,0,True,120.0,11.737,12.0,12.244,120.0
        performance,True,0,True,120.0,12.0,12.0,12.0,120.0
        random,True,0,True,120.0,2.0,12.0,102.0,120.0
        cate_optimal,True,0,True,120.0,12.0,12.0,12.0,120.0
        """
        # Run test.
        self.helper(config, cate_values, expected)


# #############################################################################
# Test_cell6_1_run_seed_sweep
# #############################################################################


class Test_cell6_1_run_seed_sweep(hunitest.TestCase):
    """
    Test the `cell6_1_run_seed_sweep()` function.
    """

    def test1(self) -> None:
        """
        Test that each row is the run of the config with that seed.
        """
        # Prepare inputs.
        config = _get_small_config()
        seeds = [1, 2]
        # Prepare outputs.
        expected_gini = []
        for seed in seeds:
            config["seed"] = seed
            results = rcaoasalcsu.run_experiment(config)
            expected_gini.append(results["stats"]["gini_coefficient"])
        # Run test.
        sweep_df = rcaoasalcsau.cell6_1_run_seed_sweep(config, seeds)
        # Check outputs.
        self.assert_equal(str(sweep_df.index.tolist()), str(seeds))
        self.assert_equal(str(sweep_df["gini"].tolist()), str(expected_gini))

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Freeze the mean of the metrics over the 50 seeds of the notebook.
        """
        # Prepare inputs.
        config = rcaoasalcsu.get_default_config()
        seeds = list(range(50))
        # Prepare outputs.
        expected = """
        ,0
        talent_std,0.1484
        gini,0.2757
        top_10_pct_share,0.224
        bottom_50_pct_share,0.312
        mean_lucky_events,2.1704
        top_decile_mean_lucky_events,4.712
        r_talent_norm_log_capital,0.1254
        r_lucky_events_log_capital,0.6518
        r_net_events_log_capital,0.9424
        """
        # Run test.
        sweep_df = rcaoasalcsau.cell6_1_run_seed_sweep(config, seeds)
        # Check outputs.
        actual = _df_to_str(sweep_df.mean().to_frame())
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_cell6_2_compare_with_paper
# #############################################################################


class Test_cell6_2_compare_with_paper(hunitest.TestCase):
    """
    Test the `cell6_2_compare_with_paper()` function.
    """

    def test1(self) -> None:
        """
        Test that the baseline-only estimates have no seed statistics.
        """
        # Prepare inputs.
        config = _get_small_config()
        baseline_results = rcaoasalcsu.run_experiment(config)
        seeds = [1, 2]
        sweep_df = rcaoasalcsau.cell6_1_run_seed_sweep(config, seeds)
        dml_effect = 0.1
        cate_std = 0.01
        # Prepare outputs.
        expected = """
        metric,paper,paper_basis,baseline,seeds_mean,seeds_std
        talent_std,0.11,results section; set-up says 0.15,0.118,0.1349,0.0119
        gini,0.38,"average over runs, number of runs not stated",0.1763,0.2406,0.0192
        top_10_pct_share,0.28,not stated,0.1722,0.2112,0.0012
        bottom_50_pct_share,0.25,not stated,0.3795,0.3322,0.0224
        mean_lucky_events,4.8,not stated,1.75,1.25,0.2121
        top_decile_mean_lucky_events,8.3,not stated,4.5,3.75,0.3536
        r_talent_norm_log_capital,0.12,not stated,0.0149,0.1044,0.0159
        r_lucky_events_log_capital,0.78,not stated,0.6972,0.7483,0.0203
        r_net_events_log_capital,0.82,not stated,0.8582,0.9612,0.0236
        dml_effect,0.124,not stated,0.1,,
        cate_std,0.03,not stated,0.01,,
        """
        # Run test.
        comparison_df = rcaoasalcsau.cell6_2_compare_with_paper(
            baseline_results, sweep_df, dml_effect, cate_std
        )
        # Check outputs.
        actual = _df_to_str(comparison_df)
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_cell6_3_plot_seed_sweep
# #############################################################################


class Test_cell6_3_plot_seed_sweep(hunitest.TestCase):
    """
    Test the `cell6_3_plot_seed_sweep()` function.
    """

    def test1(self) -> None:
        """
        Test that the panels name the metric and the number of seeds.
        """
        # Prepare inputs.
        config = _get_small_config()
        baseline_results = rcaoasalcsu.run_experiment(config)
        seeds = [1, 2]
        sweep_df = rcaoasalcsau.cell6_1_run_seed_sweep(config, seeds)
        dml_effect = 0.1
        cate_std = 0.01
        comparison_df = rcaoasalcsau.cell6_2_compare_with_paper(
            baseline_results, sweep_df, dml_effect, cate_std
        )
        setup_txt = "setup"
        # Prepare outputs.
        expected = """
        Seed variability over 2 seeds, baseline: setup
        Gini of final capital
        mean 0.241, baseline 0.176, paper 0.380
        r(lucky events, log(capital))
        mean 0.748, baseline 0.697, paper 0.780
        """
        # Run test.
        rcaoasalcsau.cell6_3_plot_seed_sweep(sweep_df, comparison_df, setup_txt)
        # Check outputs.
        actual = _get_figure_titles()
        self.assert_equal(actual, expected, dedent=True)
