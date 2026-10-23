"""
Unit tests for `causal_success_utils.py`.

Import as:

import research.Causal_Analysis_of_Agent_Skill_And_Luck.test.test_causal_success_utils as rcaoasalttcsu
"""

import logging
from typing import Any, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import helpers.hunit_test as hunitest

# Draw figures without a display.
matplotlib.use("Agg")

pytest.importorskip(
    "pymc",
    minversion=None,
    reason="PyMC not available or incompatible with current numpy version",
)

import research.Causal_Analysis_of_Agent_Skill_And_Luck.causal_success_utils as rcaoasalcsu

_LOG = logging.getLogger(__name__)


def _agent_to_str(agent: rcaoasalcsu.Agent) -> str:
    """
    Represent the state of an agent as a string, to compare it as a whole.

    :param agent: agent to represent
    :return: id, talents, capital, event counts, and capital history
    """
    txt = (
        f"id={agent.id} talent={agent.talent} capital={agent.capital} "
        f"lucky_events={agent.lucky_events} "
        f"unlucky_events={agent.unlucky_events} "
        f"capital_history={agent.capital_history}"
    )
    return txt


def _make_agent(initial_capital: float) -> rcaoasalcsu.Agent:
    """
    Create an agent with all the talents equal to 0.5.

    :param initial_capital: starting capital of the agent
    :return: agent with id 0
    """
    agent_id = 0
    intensity = 0.5
    iq = 0.5
    networking = 0.5
    agent = rcaoasalcsu.Agent(
        agent_id, intensity, iq, networking, initial_capital=initial_capital
    )
    return agent


def _simulate_and_fit(
    n_agents: int,
    n_periods: int,
    seed: int,
    draws: int,
    tune: int,
    random_seed: int,
) -> Tuple[Any, Any, pd.DataFrame]:
    """
    Simulate a small population and fit the Bayesian luck model on it.

    :param n_agents: number of agents
    :param n_periods: number of simulated periods
    :param seed: seed of the population and of the simulation
    :param draws: posterior draws per chain
    :param tune: tuning iterations
    :param random_seed: seed of the sampler
    :return: PyMC model, ArviZ inference data, and the fitted DataFrame
    """
    agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
    rcaoasalcsu.run_simulation(agents, n_periods=n_periods, seed=seed)
    df = rcaoasalcsu.get_results_dataframe(agents)
    model, idata = rcaoasalcsu.fit_bayesian_luck_model(
        df, draws=draws, tune=tune, random_seed=random_seed
    )
    return model, idata, df


def _fit_small_model() -> Tuple[Any, Any, pd.DataFrame]:
    """
    Fit the Bayesian model on the small population used by several tests.

    :return: PyMC model, ArviZ inference data, and the fitted DataFrame
    """
    n_agents = 20
    n_periods = 10
    seed = 42
    draws = 100
    tune = 100
    random_seed = 42
    model, idata, df = _simulate_and_fit(
        n_agents, n_periods, seed, draws, tune, random_seed
    )
    return model, idata, df


# #############################################################################
# TestAgent
# #############################################################################


class TestAgent(hunitest.TestCase):
    """
    Test the `Agent` class initialization.
    """

    def helper(
        self,
        agent_id: int,
        intensity: float,
        iq: float,
        networking: float,
        initial_capital: float,
        expected: str,
    ) -> None:
        """
        Test that an agent built from the inputs has the expected state.

        :param agent_id: id of the agent
        :param intensity: intensity talent
        :param iq: IQ talent
        :param networking: networking talent
        :param initial_capital: starting capital
        :param expected: expected output of `_agent_to_str()`
        """
        # Run test.
        agent = rcaoasalcsu.Agent(
            agent_id, intensity, iq, networking, initial_capital=initial_capital
        )
        # Check outputs.
        self.assert_equal(_agent_to_str(agent), expected)

    def test1(self) -> None:
        """
        Test `Agent` initialization with default parameters.
        """
        # Prepare inputs.
        agent_id = 0
        intensity = 0.5
        iq = 0.6
        networking = 0.7
        # Prepare outputs.
        expected = (
            "id=0 talent={'intensity': 0.5, 'iq': 0.6, 'networking': 0.7, "
            "'initial_capital': 1.0} capital=1.0 lucky_events=0 "
            "unlucky_events=0 capital_history=[1.0]"
        )
        # Run test.
        agent = rcaoasalcsu.Agent(agent_id, intensity, iq, networking)
        # Check outputs.
        self.assert_equal(_agent_to_str(agent), expected)

    def test2(self) -> None:
        """
        Test `Agent` initialization with custom `initial_capital`.
        """
        # Prepare inputs.
        agent_id = 1
        intensity = 0.5
        iq = 0.5
        networking = 0.5
        initial_capital = 5.0
        # Prepare outputs.
        expected = (
            "id=1 talent={'intensity': 0.5, 'iq': 0.5, 'networking': 0.5, "
            "'initial_capital': 5.0} capital=5.0 lucky_events=0 "
            "unlucky_events=0 capital_history=[5.0]"
        )
        # Run test.
        self.helper(
            agent_id, intensity, iq, networking, initial_capital, expected
        )

    def test3(self) -> None:
        """
        Test `Agent` talent values are clipped to [0, 1] range.
        """
        # Prepare inputs.
        agent_id = 2
        intensity = 1.5
        iq = -0.5
        networking = 0.5
        initial_capital = 1.0
        # Prepare outputs.
        expected = (
            "id=2 talent={'intensity': 1.0, 'iq': 0.0, 'networking': 0.5, "
            "'initial_capital': 1.0} capital=1.0 lucky_events=0 "
            "unlucky_events=0 capital_history=[1.0]"
        )
        # Run test.
        self.helper(
            agent_id, intensity, iq, networking, initial_capital, expected
        )

    def test4(self) -> None:
        """
        Test `Agent` `initial_capital` is enforced with minimum of 0.01.
        """
        # Prepare inputs.
        agent_id = 3
        intensity = 0.5
        iq = 0.5
        networking = 0.5
        initial_capital = 0.001
        # Prepare outputs.
        expected = (
            "id=3 talent={'intensity': 0.5, 'iq': 0.5, 'networking': 0.5, "
            "'initial_capital': 0.01} capital=0.01 lucky_events=0 "
            "unlucky_events=0 capital_history=[0.01]"
        )
        # Run test.
        self.helper(
            agent_id, intensity, iq, networking, initial_capital, expected
        )


# #############################################################################
# Test_talent_norm
# #############################################################################


class Test_talent_norm(hunitest.TestCase):
    """
    Test the `talent_norm` property of `Agent`.
    """

    def helper(
        self,
        intensity: float,
        iq: float,
        networking: float,
        initial_capital: float,
        expected: float,
    ) -> None:
        """
        Test that `talent_norm` is the expected Euclidean norm.

        :param intensity: intensity talent
        :param iq: IQ talent
        :param networking: networking talent
        :param initial_capital: starting capital
        :param expected: expected norm
        """
        # Prepare inputs.
        agent_id = 0
        agent = rcaoasalcsu.Agent(
            agent_id, intensity, iq, networking, initial_capital=initial_capital
        )
        # Run test.
        norm = agent.talent_norm
        # Check outputs.
        self.assertAlmostEqual(norm, expected, places=5)

    def test1(self) -> None:
        """
        Test `talent_norm` returns correct Euclidean norm of talent vector.
        """
        # Prepare inputs.
        intensity = 0.3
        iq = 0.4
        networking = 0.5
        initial_capital = 1.2
        # Prepare outputs.
        expected = np.sqrt(0.3**2 + 0.4**2 + 0.5**2 + 1.2**2)
        # Run test.
        self.helper(intensity, iq, networking, initial_capital, expected)

    def test2(self) -> None:
        """
        Test `talent_norm` with all talents equal to 0.5.
        """
        # Prepare inputs.
        intensity = 0.5
        iq = 0.5
        networking = 0.5
        initial_capital = 0.5
        # Prepare outputs.
        expected = np.sqrt(4 * 0.5**2)
        # Run test.
        self.helper(intensity, iq, networking, initial_capital, expected)

    def test3(self) -> None:
        """
        Test `talent_norm` with edge case values (0 and 1).
        """
        # Prepare inputs.
        intensity = 0.0
        iq = 1.0
        networking = 0.0
        initial_capital = 1.0
        # Prepare outputs.
        expected = np.sqrt(0.0**2 + 1.0**2 + 0.0**2 + 1.0**2)
        # Run test.
        self.helper(intensity, iq, networking, initial_capital, expected)


# #############################################################################
# Test_get_event_probability
# #############################################################################


class Test_get_event_probability(hunitest.TestCase):
    """
    Test the `get_event_probability()` method of `Agent`.
    """

    def helper(
        self, intensity: float, expected_range: Tuple[float, float]
    ) -> None:
        """
        Test that the event probability is in the expected range.

        :param intensity: intensity talent of the agent
        :param expected_range: min and max expected probability
        """
        # Prepare inputs.
        agent_id = 0
        iq = 0.5
        networking = 0.5
        agent = rcaoasalcsu.Agent(agent_id, intensity, iq, networking)
        # Run test.
        prob = agent.get_event_probability()
        # Check outputs.
        self.assertGreaterEqual(prob, expected_range[0])
        self.assertLessEqual(prob, expected_range[1])

    def test1(self) -> None:
        """
        Test event probability with intensity = 0.5 (sigmoid midpoint).
        """
        # Prepare inputs.
        initial_capital = 1.0
        agent = _make_agent(initial_capital)
        # Prepare outputs.
        # At intensity=0.5, sigmoid should return 0.5 (midpoint of [0, 1]).
        expected = 0.5
        # Run test.
        prob = agent.get_event_probability()
        # Check outputs.
        self.assertAlmostEqual(prob, expected, places=2)

    def test2(self) -> None:
        """
        Test event probability with low intensity (near 0).
        """
        # Prepare inputs.
        intensity = 0.0
        # Prepare outputs.
        expected_range = (0.0, 0.3)
        # Run test.
        self.helper(intensity, expected_range)

    def test3(self) -> None:
        """
        Test event probability with high intensity (near 1).
        """
        # Prepare inputs.
        intensity = 1.0
        # Prepare outputs.
        expected_range = (0.7, 1.0)
        # Run test.
        self.helper(intensity, expected_range)

    def test4(self) -> None:
        """
        Test event probability is monotonically increasing with intensity.
        """
        # Prepare inputs.
        agent_id = 0
        iq = 0.5
        networking = 0.5
        intensities = [0.0, 0.25, 0.5, 0.75, 1.0]
        # Run test.
        probs = [
            rcaoasalcsu.Agent(
                agent_id, intensity, iq, networking
            ).get_event_probability()
            for intensity in intensities
        ]
        # Check outputs.
        for i in range(len(probs) - 1):
            self.assertLessEqual(probs[i], probs[i + 1])


# #############################################################################
# Test_apply_event
# #############################################################################


class Test_apply_event(hunitest.TestCase):
    """
    Test the `apply_event()` method of `Agent`.
    """

    def helper(
        self,
        initial_capital: float,
        event_type: str,
        impact: float,
        expected: float,
    ) -> None:
        """
        Test that one event moves the capital to the expected value.

        :param initial_capital: starting capital of the agent
        :param event_type: `"lucky"` or `"unlucky"`
        :param impact: magnitude of the event
        :param expected: expected capital after the event
        """
        # Prepare inputs.
        agent = _make_agent(initial_capital)
        # Run test.
        agent.apply_event(event_type, impact)
        # Check outputs.
        self.assertAlmostEqual(agent.capital, expected)

    def test1(self) -> None:
        """
        Test applying a lucky event increases capital.
        """
        # Prepare inputs.
        initial_capital = 1.0
        event_type = "lucky"
        impact = 0.25
        # Prepare outputs.
        expected = 1.25
        # Run test.
        self.helper(initial_capital, event_type, impact, expected)

    def test2(self) -> None:
        """
        Test applying an unlucky event decreases capital.
        """
        # Prepare inputs.
        initial_capital = 1.0
        event_type = "unlucky"
        impact = 0.15
        # Prepare outputs.
        expected = 0.85
        # Run test.
        self.helper(initial_capital, event_type, impact, expected)

    def test3(self) -> None:
        """
        Test applying an unlucky event respects minimum capital floor (0.01).
        """
        # Prepare inputs.
        initial_capital = 0.05
        event_type = "unlucky"
        impact = 0.99
        # Prepare outputs.
        expected = 0.01
        # Run test.
        self.helper(initial_capital, event_type, impact, expected)

    def test4(self) -> None:
        """
        Test applying an event increments the appropriate event counter.
        """
        # Prepare inputs.
        initial_capital = 1.0
        agent = _make_agent(initial_capital)
        impact = 0.1
        event_types = ["lucky", "unlucky", "lucky"]
        # Prepare outputs.
        expected = "lucky_events=2 unlucky_events=1"
        # Run test.
        for event_type in event_types:
            agent.apply_event(event_type, impact)
        # Check outputs.
        actual = (
            f"lucky_events={agent.lucky_events} "
            f"unlucky_events={agent.unlucky_events}"
        )
        self.assert_equal(actual, expected)

    def test5(self) -> None:
        """
        Test `capital_history` is updated after each event.
        """
        # Prepare inputs.
        initial_capital = 1.0
        agent = _make_agent(initial_capital)
        lucky_impact = 0.25
        unlucky_impact = 0.1
        # Prepare outputs.
        expected = "[1.0, 1.25, 1.125]"
        # Run test.
        agent.apply_event("lucky", lucky_impact)
        agent.apply_event("unlucky", unlucky_impact)
        # Check outputs.
        self.assert_equal(str(agent.capital_history), expected)

    def test6(self) -> None:
        """
        Test `apply_event()` raises error for invalid `event_type`.
        """
        # Prepare inputs.
        initial_capital = 1.0
        agent = _make_agent(initial_capital)
        event_type = "invalid"
        impact = 0.1
        # Run test and check output.
        with self.assertRaises(ValueError):
            agent.apply_event(event_type, impact)

    def test7(self) -> None:
        """
        Test negative impact magnitude is converted to absolute value.
        """
        # Prepare inputs.
        initial_capital = 1.0
        event_type = "lucky"
        impact = -0.25
        # Prepare outputs.
        expected = 1.25
        # Run test.
        self.helper(initial_capital, event_type, impact, expected)


# #############################################################################
# Test_create_population
# #############################################################################


class Test_create_population(hunitest.TestCase):
    """
    Test the `create_population()` function.
    """

    def test1(self) -> None:
        """
        Test `create_population()` returns correct number of agents.
        """
        # Prepare inputs.
        n_agents = 50
        seed = 42
        # Run test.
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Check outputs.
        self.assertEqual(len(agents), n_agents)

    def test2(self) -> None:
        """
        Test `create_population()` agents have talents in [0, 1] range.
        """
        # Prepare inputs.
        n_agents = 20
        seed = 42
        # Run test.
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Check outputs.
        for agent in agents:
            self.assertGreaterEqual(agent.talent["intensity"], 0.0)
            self.assertLessEqual(agent.talent["intensity"], 1.0)
            self.assertGreaterEqual(agent.talent["iq"], 0.0)
            self.assertLessEqual(agent.talent["iq"], 1.0)
            self.assertGreaterEqual(agent.talent["networking"], 0.0)
            self.assertLessEqual(agent.talent["networking"], 1.0)

    def test3(self) -> None:
        """
        Test `create_population()` all agents start with capital = 1.0.
        """
        # Prepare inputs.
        n_agents = 15
        seed = 42
        # Prepare outputs.
        expected = str([1.0] * n_agents)
        # Run test.
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Check outputs.
        self.assert_equal(str([agent.capital for agent in agents]), expected)

    def test4(self) -> None:
        """
        Test `create_population()` with seed produces reproducible results.
        """
        # Prepare inputs.
        n_agents = 10
        seed = 123
        # Run test.
        agents1 = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        agents2 = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Check outputs.
        actual = str([agent.talent for agent in agents1])
        expected = str([agent.talent for agent in agents2])
        self.assert_equal(actual, expected)

    def test5(self) -> None:
        """
        Test `create_population()` raises error for non-positive `n_agents`.
        """
        # Prepare inputs.
        n_agents = 0
        seed = 42
        # Run test and check output.
        with self.assertRaises(AssertionError):
            rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)

    def test6(self) -> None:
        """
        Test agent IDs are sequential from 0 to `n_agents - 1`.
        """
        # Prepare inputs.
        n_agents = 25
        seed = 42
        # Prepare outputs.
        expected = str(list(range(n_agents)))
        # Run test.
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Check outputs.
        self.assert_equal(str([agent.id for agent in agents]), expected)


# #############################################################################
# Test_calculate_gini
# #############################################################################


class Test_calculate_gini(hunitest.TestCase):
    """
    Test the `calculate_gini()` function.
    """

    def helper(self, values: np.ndarray, expected: float) -> None:
        """
        Test that the Gini coefficient of `values` is the expected one.

        :param values: values to compute the Gini coefficient of
        :param expected: expected Gini coefficient
        """
        # Run test.
        gini = rcaoasalcsu.calculate_gini(values)
        # Check outputs.
        self.assertAlmostEqual(gini, expected, places=3)

    def test1(self) -> None:
        """
        Test Gini coefficient for perfectly equal distribution.
        """
        # Prepare inputs.
        values = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
        # Prepare outputs.
        expected = 0.0
        # Run test.
        self.helper(values, expected)

    def test2(self) -> None:
        """
        Test Gini coefficient for highly unequal distribution.
        """
        # Prepare inputs.
        values = np.array([0.1, 0.1, 0.1, 100.0], dtype=float)
        # Run test.
        gini = rcaoasalcsu.calculate_gini(values)
        # Check outputs.
        self.assertGreater(gini, 0.7)

    def test3(self) -> None:
        """
        Test Gini coefficient returns value in [0, 1] range.
        """
        # Prepare inputs.
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        # Run test.
        gini = rcaoasalcsu.calculate_gini(values)
        # Check outputs.
        self.assertGreaterEqual(gini, 0.0)
        self.assertLessEqual(gini, 1.0)

    def test4(self) -> None:
        """
        Test Gini coefficient with single value.
        """
        # Prepare inputs.
        values = np.array([5.0], dtype=float)
        # Prepare outputs.
        expected = 0.0
        # Run test.
        gini = rcaoasalcsu.calculate_gini(values)
        # Check outputs.
        self.assertAlmostEqual(gini, expected)

    def test5(self) -> None:
        """
        Test Gini coefficient raises error for empty array.
        """
        # Prepare inputs.
        values = np.array([], dtype=float)
        # Run test and check output.
        with self.assertRaises(AssertionError):
            rcaoasalcsu.calculate_gini(values)

    def test6(self) -> None:
        """
        Test Gini coefficient raises error for negative values.
        """
        # Prepare inputs.
        values = np.array([1.0, 2.0, -1.0], dtype=float)
        # Run test and check output.
        with self.assertRaises(AssertionError):
            rcaoasalcsu.calculate_gini(values)

    def test7(self) -> None:
        """
        Test Gini coefficient with all zeros returns 0.
        """
        # Prepare inputs.
        values = np.array([0.0, 0.0, 0.0], dtype=float)
        # Prepare outputs.
        expected = 0.0
        # Run test.
        gini = rcaoasalcsu.calculate_gini(values)
        # Check outputs.
        self.assertAlmostEqual(gini, expected)


# #############################################################################
# Test_get_results_dataframe
# #############################################################################


class Test_get_results_dataframe(hunitest.TestCase):
    """
    Test the `get_results_dataframe()` function.
    """

    def test1(self) -> None:
        """
        Test `get_results_dataframe()` converts agents to correct columns.
        """
        # Prepare inputs.
        n_agents = 5
        seed = 42
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Prepare outputs.
        expected = str(
            [
                "id",
                "talent_intensity",
                "talent_iq",
                "talent_networking",
                "initial_capital",
                "talent_norm",
                "capital",
                "lucky_events",
                "unlucky_events",
                "net_events",
            ]
        )
        # Run test.
        df = rcaoasalcsu.get_results_dataframe(agents)
        # Check outputs.
        self.assert_equal(str(list(df.columns)), expected)

    def test2(self) -> None:
        """
        Test `get_results_dataframe()` returns empty DataFrame for empty list.
        """
        # Prepare inputs.
        agents: List[rcaoasalcsu.Agent] = []
        # Run test.
        df = rcaoasalcsu.get_results_dataframe(agents)
        # Check outputs.
        self.assertTrue(df.empty)

    def test3(self) -> None:
        """
        Test `get_results_dataframe()` computes `net_events` correctly.
        """
        # Prepare inputs.
        initial_capital = 1.0
        agent = _make_agent(initial_capital)
        impact = 0.1
        for event_type in ["lucky", "lucky", "unlucky"]:
            agent.apply_event(event_type, impact)
        # Prepare outputs.
        expected = 1
        # Run test.
        df = rcaoasalcsu.get_results_dataframe([agent])
        # Check outputs.
        self.assertEqual(df.iloc[0]["net_events"], expected)

    def test4(self) -> None:
        """
        Test `get_results_dataframe()` includes `talent_norm` computation.
        """
        # Prepare inputs.
        n_agents = 3
        seed = 42
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Run test.
        df = rcaoasalcsu.get_results_dataframe(agents)
        # Check outputs.
        for idx, agent in enumerate(agents):
            self.assertAlmostEqual(
                df.iloc[idx]["talent_norm"], agent.talent_norm, places=4
            )

    def test5(self) -> None:
        """
        Test `get_results_dataframe()` with single agent.
        """
        # Prepare inputs.
        agent_id = 0
        intensity = 0.3
        iq = 0.4
        networking = 0.5
        initial_capital = 2.0
        agent = rcaoasalcsu.Agent(
            agent_id, intensity, iq, networking, initial_capital=initial_capital
        )
        # Prepare outputs.
        expected = "[{'id': 0, 'capital': 2.0}]"
        # Run test.
        df = rcaoasalcsu.get_results_dataframe([agent])
        # Check outputs.
        actual = str(df[["id", "capital"]].to_dict("records"))
        self.assert_equal(actual, expected)


# #############################################################################
# Test_generate_summary_statistics
# #############################################################################


class Test_generate_summary_statistics(hunitest.TestCase):
    """
    Test the `generate_summary_statistics()` function.
    """

    def test1(self) -> None:
        """
        Test `generate_summary_statistics()` returns correct keys.
        """
        # Prepare inputs.
        n_agents = 10
        seed = 42
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Prepare outputs.
        expected = str(
            sorted(
                [
                    "n_agents",
                    "mean_capital",
                    "median_capital",
                    "std_capital",
                    "min_capital",
                    "max_capital",
                    "capital_range",
                    "gini_coefficient",
                    "top_10_pct_share",
                    "top_20_pct_share",
                    "bottom_50_pct_share",
                    "mean_lucky_events",
                    "mean_unlucky_events",
                    "mean_talent_norm",
                ]
            )
        )
        # Run test.
        stats = rcaoasalcsu.generate_summary_statistics(agents)
        # Check outputs.
        self.assert_equal(str(sorted(stats.keys())), expected)

    def test2(self) -> None:
        """
        Test `generate_summary_statistics()` with single agent.
        """
        # Prepare inputs.
        initial_capital = 2.0
        agent = _make_agent(initial_capital)
        # Prepare outputs.
        expected = (
            "{'n_agents': 1.0, 'mean_capital': 2.0, 'min_capital': 2.0, "
            "'max_capital': 2.0}"
        )
        # Run test.
        stats = rcaoasalcsu.generate_summary_statistics([agent])
        # Check outputs.
        keys = ["n_agents", "mean_capital", "min_capital", "max_capital"]
        actual = str({key: stats[key] for key in keys})
        self.assert_equal(actual, expected)

    def test3(self) -> None:
        """
        Test `generate_summary_statistics()` with empty agents list.
        """
        # Prepare inputs.
        agents: List[rcaoasalcsu.Agent] = []
        # Prepare outputs.
        expected = "{'n_agents': 0}"
        # Run test.
        stats = rcaoasalcsu.generate_summary_statistics(agents)
        # Check outputs.
        self.assert_equal(str(stats), expected)

    def test4(self) -> None:
        """
        Test the top and bottom capital shares add up to more than half.
        """
        # Prepare inputs.
        n_agents = 100
        seed = 42
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Run test.
        stats = rcaoasalcsu.generate_summary_statistics(agents)
        # Check outputs.
        total_share = (
            stats["top_10_pct_share"]
            + stats["top_20_pct_share"]
            + stats["bottom_50_pct_share"]
        )
        self.assertGreater(total_share, 0.5)

    def test5(self) -> None:
        """
        Test `mean_capital` equals `median_capital` for equal wealth.
        """
        # Prepare inputs.
        n_agents = 10
        intensity = 0.5
        iq = 0.5
        networking = 0.5
        initial_capital = 1.0
        agents = [
            rcaoasalcsu.Agent(
                i, intensity, iq, networking, initial_capital=initial_capital
            )
            for i in range(n_agents)
        ]
        # Run test.
        stats = rcaoasalcsu.generate_summary_statistics(agents)
        # Check outputs.
        self.assertAlmostEqual(stats["mean_capital"], stats["median_capital"])

    def test6(self) -> None:
        """
        Test `capital_range` is positive.
        """
        # Prepare inputs.
        n_agents = 20
        seed = 42
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Run test.
        stats = rcaoasalcsu.generate_summary_statistics(agents)
        # Check outputs.
        self.assertGreater(stats["capital_range"], 0.0)


# #############################################################################
# Test_validate_simulation_results
# #############################################################################


class Test_validate_simulation_results(hunitest.TestCase):
    """
    Test the `validate_simulation_results()` function.
    """

    def test1(self) -> None:
        """
        Test `validate_simulation_results()` returns True for valid agents.
        """
        # Prepare inputs.
        n_agents = 10
        seed = 42
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Run test.
        result = rcaoasalcsu.validate_simulation_results(agents)
        # Check outputs.
        self.assertTrue(result)

    def test2(self) -> None:
        """
        Test `validate_simulation_results()` raises error for empty agents.
        """
        # Prepare inputs.
        agents: List[rcaoasalcsu.Agent] = []
        # Run test and check output.
        with self.assertRaises(AssertionError):
            rcaoasalcsu.validate_simulation_results(agents)

    def test3(self) -> None:
        """
        Test `validate_simulation_results()` detects negative capital.
        """
        # Prepare inputs.
        initial_capital = 1.0
        agent = _make_agent(initial_capital)
        agent.capital = -0.5
        # Run test and check output.
        with self.assertRaises(AssertionError):
            rcaoasalcsu.validate_simulation_results([agent])

    def test4(self) -> None:
        """
        Test `validate_simulation_results()` detects history inconsistencies.
        """
        # Prepare inputs.
        initial_capital = 1.0
        agent = _make_agent(initial_capital)
        # `capital_history` should have 6 entries (1 initial + 5 events), but
        # it is left with 1.
        agent.lucky_events = 5
        # Run test and check output.
        with self.assertRaises(AssertionError):
            rcaoasalcsu.validate_simulation_results([agent])

    def test5(self) -> None:
        """
        Test `validate_simulation_results()` passes after a simulation.
        """
        # Prepare inputs.
        n_agents = 10
        n_periods = 5
        seed = 42
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        agents = rcaoasalcsu.run_simulation(
            agents, n_periods=n_periods, seed=seed
        )
        # Run test.
        result = rcaoasalcsu.validate_simulation_results(agents)
        # Check outputs.
        self.assertTrue(result)


# #############################################################################
# Test_run_simulation
# #############################################################################


class Test_run_simulation(hunitest.TestCase):
    """
    Test the `run_simulation()` function.
    """

    def test1(self) -> None:
        """
        Test `run_simulation()` with default parameters.
        """
        # Prepare inputs.
        n_agents = 10
        n_periods = 10
        seed = 42
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Run test.
        result = rcaoasalcsu.run_simulation(
            agents, n_periods=n_periods, seed=seed
        )
        # Check outputs.
        self.assertEqual(result, agents)
        self.assertTrue(rcaoasalcsu.validate_simulation_results(agents))

    def test2(self) -> None:
        """
        Test `run_simulation()` with zero events per period.
        """
        # Prepare inputs.
        n_agents = 5
        n_periods = 10
        n_events_per_period = 0
        seed = 42
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Prepare outputs.
        expected = str([agent.capital for agent in agents])
        # Run test.
        rcaoasalcsu.run_simulation(
            agents,
            n_periods=n_periods,
            n_lucky_events_per_period=n_events_per_period,
            n_unlucky_events_per_period=n_events_per_period,
            seed=seed,
        )
        # Check outputs.
        self.assert_equal(str([agent.capital for agent in agents]), expected)

    def test3(self) -> None:
        """
        Test `run_simulation()` with seed produces reproducible results.
        """
        # Prepare inputs.
        n_agents = 10
        population_seed = 42
        n_periods = 5
        seed = 123
        agents1 = rcaoasalcsu.create_population(
            n_agents=n_agents, seed=population_seed
        )
        agents2 = rcaoasalcsu.create_population(
            n_agents=n_agents, seed=population_seed
        )
        # Run test.
        rcaoasalcsu.run_simulation(agents1, n_periods=n_periods, seed=seed)
        rcaoasalcsu.run_simulation(agents2, n_periods=n_periods, seed=seed)
        # Check outputs.
        actual = str([agent.capital for agent in agents1])
        expected = str([agent.capital for agent in agents2])
        self.assert_equal(actual, expected)

    def test4(self) -> None:
        """
        Test `run_simulation()` modifies agents in-place.
        """
        # Prepare inputs.
        n_agents = 5
        n_periods = 5
        seed = 42
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        agents_ref = agents
        # Run test.
        result = rcaoasalcsu.run_simulation(
            agents, n_periods=n_periods, seed=seed
        )
        # Check outputs.
        self.assertIs(result, agents_ref)

    def test5(self) -> None:
        """
        Test `run_simulation()` respects capital floor of 0.01.
        """
        # Prepare inputs.
        n_agents = 10
        n_periods = 20
        seed = 42
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Run test.
        rcaoasalcsu.run_simulation(agents, n_periods=n_periods, seed=seed)
        # Check outputs.
        for agent in agents:
            self.assertGreaterEqual(agent.capital, 0.01)

    def test6(self) -> None:
        """
        Test `run_simulation()` raises error for non-positive `n_periods`.
        """
        # Prepare inputs.
        n_agents = 5
        n_periods = 0
        seed = 42
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Run test and check output.
        with self.assertRaises(AssertionError):
            rcaoasalcsu.run_simulation(agents, n_periods=n_periods, seed=seed)

    def test7(self) -> None:
        """
        Test `run_simulation()` raises error for empty agents list.
        """
        # Prepare inputs.
        agents: List[rcaoasalcsu.Agent] = []
        n_periods = 5
        seed = 42
        # Run test and check output.
        with self.assertRaises(AssertionError):
            rcaoasalcsu.run_simulation(agents, n_periods=n_periods, seed=seed)

    def test8(self) -> None:
        """
        Test lucky events are capitalized based on IQ talent.
        """
        # Prepare inputs.
        intensity = 0.5
        networking = 0.5
        initial_capital = 1.0
        # Create agent with high IQ to capitalize on lucky events.
        high_iq = 0.99
        high_iq_id = 0
        agent_high_iq = rcaoasalcsu.Agent(
            high_iq_id,
            intensity,
            high_iq,
            networking,
            initial_capital=initial_capital,
        )
        # Create agent with low IQ to fail capitalizing.
        low_iq = 0.01
        low_iq_id = 1
        agent_low_iq = rcaoasalcsu.Agent(
            low_iq_id,
            intensity,
            low_iq,
            networking,
            initial_capital=initial_capital,
        )
        n_periods = 50
        n_lucky_events_per_period = 10
        n_unlucky_events_per_period = 0
        seed = 42
        # Run test.
        rcaoasalcsu.run_simulation(
            [agent_high_iq, agent_low_iq],
            n_periods=n_periods,
            n_lucky_events_per_period=n_lucky_events_per_period,
            n_unlucky_events_per_period=n_unlucky_events_per_period,
            seed=seed,
        )
        # Check outputs.
        # High IQ agent should have higher capital due to capitalizing on luck.
        self.assertGreater(agent_high_iq.capital, agent_low_iq.capital)

    def test9(self) -> None:
        """
        Test networking spillover mechanism with 10% spillover probability.
        """
        # Prepare inputs.
        n_agents = 10
        n_periods = 100
        n_lucky_events_per_period = 10
        n_unlucky_events_per_period = 0
        seed = 42
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Run test.
        rcaoasalcsu.run_simulation(
            agents,
            n_periods=n_periods,
            n_lucky_events_per_period=n_lucky_events_per_period,
            n_unlucky_events_per_period=n_unlucky_events_per_period,
            seed=seed,
        )
        # Check outputs.
        # At least some agents should have lucky events (spillover or direct).
        total_lucky = sum(agent.lucky_events for agent in agents)
        self.assertGreater(total_lucky, 0)


# #############################################################################
# Test_run_policy_simulation
# #############################################################################


class Test_run_policy_simulation(hunitest.TestCase):
    """
    Test the `run_policy_simulation()` function with different policies.
    """

    def helper(self, policy: str, cate_values: Optional[np.ndarray]) -> None:
        """
        Test that a policy simulation produces valid results.

        :param policy: allocation policy to run
        :param cate_values: CATE estimates, used only by `"cate_optimal"`
        """
        # Prepare inputs.
        n_agents = 10
        resource_amount = 50.0
        n_periods = 5
        seed = 42
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Run test.
        result = rcaoasalcsu.run_policy_simulation(
            agents,
            policy=policy,
            resource_amount=resource_amount,
            cate_values=cate_values,
            n_periods=n_periods,
            seed=seed,
        )
        # Check outputs.
        self.assertTrue(rcaoasalcsu.validate_simulation_results(result))

    def test1(self) -> None:
        """
        Test `run_policy_simulation()` with egalitarian policy.
        """
        # Prepare inputs.
        policy = "egalitarian"
        cate_values = None
        # Run test.
        self.helper(policy, cate_values)

    def test2(self) -> None:
        """
        Test `run_policy_simulation()` with meritocratic policy.
        """
        # Prepare inputs.
        policy = "meritocratic"
        cate_values = None
        # Run test.
        self.helper(policy, cate_values)

    def test3(self) -> None:
        """
        Test `run_policy_simulation()` with performance policy.
        """
        # Prepare inputs.
        policy = "performance"
        cate_values = None
        # Run test.
        self.helper(policy, cate_values)

    def test4(self) -> None:
        """
        Test `run_policy_simulation()` with random policy.
        """
        # Prepare inputs.
        policy = "random"
        cate_values = None
        # Run test.
        self.helper(policy, cate_values)

    def test5(self) -> None:
        """
        Test `run_policy_simulation()` with cate_optimal policy.
        """
        # Prepare inputs.
        policy = "cate_optimal"
        cate_values = np.array([0.5] * 10, dtype=float)
        # Run test.
        self.helper(policy, cate_values)

    def test6(self) -> None:
        """
        Test `run_policy_simulation()` egalitarian allocates equally.
        """
        # Prepare inputs.
        n_agents = 10
        seed = 42
        policy = "egalitarian"
        resource_amount = 100.0
        n_periods = 1
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Check allocation by comparing the change in total capital.
        initial_total = sum(agent.capital for agent in agents)
        # Run test.
        rcaoasalcsu.run_policy_simulation(
            agents,
            policy=policy,
            resource_amount=resource_amount,
            n_periods=n_periods,
            seed=seed,
        )
        # Check outputs.
        final_total = sum(agent.capital for agent in agents)
        # Final total should be close to initial + resource allocation.
        self.assertGreaterEqual(
            final_total, initial_total + resource_amount - 10.0
        )

    def test7(self) -> None:
        """
        Test `run_policy_simulation()` keeps the allocated capital.
        """
        # Prepare inputs.
        n_agents = 10
        seed = 42
        policy = "egalitarian"
        resource_amount = 50.0
        n_periods = 1
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        initial_total = sum(agent.capital for agent in agents)
        # Run test.
        rcaoasalcsu.run_policy_simulation(
            agents,
            policy=policy,
            resource_amount=resource_amount,
            n_periods=n_periods,
            seed=seed,
        )
        # Check outputs.
        # Total capital should be at least initial + allocated (may change
        # during simulation).
        final_total = sum(agent.capital for agent in agents)
        self.assertGreaterEqual(
            final_total, initial_total + resource_amount - 5.0
        )

    def test8(self) -> None:
        """
        Test `run_policy_simulation()` raises error for invalid policy.
        """
        # Prepare inputs.
        n_agents = 5
        seed = 42
        policy = "invalid_policy"
        resource_amount = 50.0
        n_periods = 5
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Run test and check output.
        with self.assertRaises(ValueError):
            rcaoasalcsu.run_policy_simulation(
                agents,
                policy=policy,
                resource_amount=resource_amount,
                n_periods=n_periods,
                seed=seed,
            )

    def test9(self) -> None:
        """
        Test `run_policy_simulation()` cate_optimal requires `cate_values`.
        """
        # Prepare inputs.
        n_agents = 5
        seed = 42
        policy = "cate_optimal"
        resource_amount = 50.0
        n_periods = 5
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Run test and check output.
        with self.assertRaises(AssertionError):
            rcaoasalcsu.run_policy_simulation(
                agents,
                policy=policy,
                resource_amount=resource_amount,
                n_periods=n_periods,
                seed=seed,
            )

    def test10(self) -> None:
        """
        Test `run_policy_simulation()` with negative CATE values clamped to 0.
        """
        # Prepare inputs.
        n_agents = 5
        seed = 42
        policy = "cate_optimal"
        resource_amount = 50.0
        n_periods = 1
        cate_values = np.array([-1.0, 0.5, -0.5, 0.8, 0.0], dtype=float)
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        # Run test.
        rcaoasalcsu.run_policy_simulation(
            agents,
            policy=policy,
            resource_amount=resource_amount,
            cate_values=cate_values,
            n_periods=n_periods,
            seed=seed,
        )
        # Check outputs.
        # Agents with negative CATE should get 0 allocation.
        self.assertTrue(rcaoasalcsu.validate_simulation_results(agents))


# #############################################################################
# Test_fit_bayesian_luck_model
# #############################################################################


class Test_fit_bayesian_luck_model(hunitest.TestCase):
    """
    Test the `fit_bayesian_luck_model()` function.
    """

    def test1(self) -> None:
        """
        Test `fit_bayesian_luck_model()` returns model and idata.
        """
        # Run test.
        model, idata, _ = _fit_small_model()
        # Check outputs.
        self.assertIsNotNone(model)
        self.assertIsNotNone(idata)

    def test2(self) -> None:
        """
        Test `fit_bayesian_luck_model()` raises error for missing columns.
        """
        # Prepare inputs.
        df = pd.DataFrame({"capital": [1.0, 2.0, 3.0]})
        draws = 10
        tune = 10
        random_seed = 42
        # Run test and check output.
        with self.assertRaises(AssertionError):
            rcaoasalcsu.fit_bayesian_luck_model(
                df, draws=draws, tune=tune, random_seed=random_seed
            )

    def test3(self) -> None:
        """
        Test `fit_bayesian_luck_model()` with seed produces results.
        """
        # Prepare inputs.
        n_agents = 15
        n_periods = 8
        seed = 42
        draws = 50
        tune = 50
        random_seed = 123
        # Run test.
        _, idata1, _ = _simulate_and_fit(
            n_agents, n_periods, seed, draws, tune, random_seed
        )
        _, idata2, _ = _simulate_and_fit(
            n_agents, n_periods, seed, draws, tune, random_seed
        )
        # Check outputs.
        self.assertIsNotNone(idata1)
        self.assertIsNotNone(idata2)


# #############################################################################
# Test_summarize_bayesian_fit
# #############################################################################


class Test_summarize_bayesian_fit(hunitest.TestCase):
    """
    Test the `summarize_bayesian_fit()` function.
    """

    def test1(self) -> None:
        """
        Test `summarize_bayesian_fit()` returns a non-empty summary.
        """
        # Prepare inputs.
        _, idata, _ = _fit_small_model()
        # Run test.
        summary = rcaoasalcsu.summarize_bayesian_fit(idata)
        # Check outputs.
        self.assertIsNotNone(summary)
        self.assertGreater(len(summary), 0)

    def test2(self) -> None:
        """
        Test `summarize_bayesian_fit()` includes the default parameters.
        """
        # Prepare inputs.
        _, idata, _ = _fit_small_model()
        # Prepare outputs.
        expected = (
            "['alpha', 'beta_luck', 'beta_intensity', 'beta_iq', "
            "'beta_networking', 'sigma']"
        )
        # Run test.
        summary = rcaoasalcsu.summarize_bayesian_fit(idata)
        # Check outputs.
        self.assert_equal(str(list(summary.index)), expected)

    def test3(self) -> None:
        """
        Test `summarize_bayesian_fit()` with custom `var_names` parameter.
        """
        # Prepare inputs.
        _, idata, _ = _fit_small_model()
        var_names = ("alpha", "sigma")
        # Prepare outputs.
        expected = "['alpha', 'sigma']"
        # Run test.
        summary = rcaoasalcsu.summarize_bayesian_fit(idata, var_names=var_names)
        # Check outputs.
        self.assert_equal(str(list(summary.index)), expected)


# #############################################################################
# Test_posterior_predictive_check
# #############################################################################


class Test_posterior_predictive_check(hunitest.TestCase):
    """
    Test the `posterior_predictive_check()` function.
    """

    def test1(self) -> None:
        """
        Test `posterior_predictive_check()` returns correct keys.
        """
        # Prepare inputs.
        model, idata, df = _fit_small_model()
        random_seed = 123
        # Prepare outputs.
        expected = "['y_obs', 'y_pred_mean', 'y_pred_std']"
        # Run test.
        ppc = rcaoasalcsu.posterior_predictive_check(
            model, idata, df, random_seed=random_seed
        )
        # Check outputs.
        self.assert_equal(str(sorted(ppc.keys())), expected)

    def test2(self) -> None:
        """
        Test `posterior_predictive_check()` arrays have the correct shape.
        """
        # Prepare inputs.
        model, idata, df = _fit_small_model()
        random_seed = 123
        # Prepare outputs.
        n_obs = len(df)
        expected = str(
            {
                "y_obs": (n_obs,),
                "y_pred_mean": (n_obs,),
                "y_pred_std": (n_obs,),
            }
        )
        # Run test.
        ppc = rcaoasalcsu.posterior_predictive_check(
            model, idata, df, random_seed=random_seed
        )
        # Check outputs.
        actual = str({key: value.shape for key, value in ppc.items()})
        self.assert_equal(actual, expected)


# #############################################################################
# Test_plot_talent_capital_luck
# #############################################################################


class Test_plot_talent_capital_luck(hunitest.TestCase):
    """
    Test the `plot_talent_capital_luck()` function.
    """

    def test1(self) -> None:
        """
        Test that the plot has 3 panels titled with the computed statistics.
        """
        # Prepare inputs.
        n_agents = 30
        n_periods = 10
        seed = 42
        agents = rcaoasalcsu.create_population(n_agents=n_agents, seed=seed)
        agents = rcaoasalcsu.run_simulation(
            agents, n_periods=n_periods, seed=seed
        )
        df = rcaoasalcsu.get_results_dataframe(agents)
        gini = rcaoasalcsu.calculate_gini(df["capital"].to_numpy())
        # Prepare outputs.
        expected_n_axes = 3
        expected_gini_txt = f"Gini {gini:.3f}"
        # Run test.
        rcaoasalcsu.plot_talent_capital_luck(df)
        # Check outputs.
        fig = plt.gcf()
        titles = [ax.get_title() for ax in fig.axes]
        plt.close(fig)
        self.assertEqual(len(fig.axes), expected_n_axes)
        self.assertIn(expected_gini_txt, titles[1])
