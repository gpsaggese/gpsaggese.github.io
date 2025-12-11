"""
template_utils.py

This file contains utility functions that support the tutorial notebooks.

- Notebooks should call these functions instead of writing raw logic inline.
- This helps keep the notebooks clean, modular, and easier to debug.
- Students should implement functions here for data preprocessing,
  model setup, evaluation, or any reusable logic.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ax import Client, RangeParameterConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definition of the Hartmann function for the API example

def hartmann6(x1, x2, x3, x4, x5, x6):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    P = 10**-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])

    outer = 0.0
    for i in range(4):
        inner = 0.0
        for j, x in enumerate([x1, x2, x3, x4, x5, x6]):
            inner += A[i, j] * (x - P[i, j])**2
        outer += alpha[i] * np.exp(-inner)
    return -outer

class MultiArmedBanditAlgorithm(ABC):
    """Abstract class to be implemented by the A/B Testing, UCB1, Thompson Sampling, and GP-Bandit algorithms."""

    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.total_reward = 0
        self.total_trials = 0

    @abstractmethod
    def select_arm(self) -> int:
        """Each algorithm has its own criteria to select the arm to pull next.
        A/B Testing: all arms have the same probability
        UCB1: arm with the highest expected reward
        Thompson Sampling: Each arm has a probability of being chosen (The probability is obtained from the Beta distribution)
        GP-Bandit: Each arm has a probability of being chosen. The probability is returned by Ax"""

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        """The reward (If the click was successful or not) is used to update the algorithm.
        A/B Testing: no update
        UCB1: update the expected reward of the arm
        Thompson Sampling: update the Beta distribution
        GP-Bandit: Accumulate the results and after certain number of pulls, update the GP model"""
        self.total_reward += reward
        self.total_trials += 1

class AB_Testing(MultiArmedBanditAlgorithm):
    def __init__(self, n_arms: int):
        super().__init__(n_arms)

    def select_arm(self) -> int:
        return np.random.randint(0, self.n_arms)

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)

class UCB1(MultiArmedBanditAlgorithm):
    def __init__(self, n_arms: int, kappa: float = 1.41):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms)
        self.n_pulls = np.zeros(n_arms)
        self.total_pulls = 0
        self.kappa = kappa

    def select_arm(self) -> int:
        # Calculate the UCB for all the arms and select the arm with the highest UCB
        ucb = np.empty(self.n_arms)
        for i in range(self.n_arms):
            # If the arm has not been pulled yet, set the UCB to infinity to avoid selecting it
            if self.n_pulls[i] == 0:
                ucb[i] = np.inf
            else:
                # Calculate the UCB for the arm
                ucb[i] = self.expected_rewards[i] + self.kappa * np.sqrt(2 * np.log(self.total_pulls) / self.n_pulls[i])
        return np.argmax(ucb)

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        # Recalculate the mean
        self.expected_rewards[arm] = (self.expected_rewards[arm] * self.n_pulls[arm] + reward) / (self.n_pulls[arm] + 1)
        self.n_pulls[arm] += 1
        self.total_pulls += 1

class ThompsonSampling(MultiArmedBanditAlgorithm):
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        # Each arm has a Beta distribution, starting with 1,1
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self) -> int:
        # Sample a value from the Beta distribution for each arm and select the arm with the highest value
        return np.argmax(np.random.beta(self.alpha, self.beta))

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        # Update the Beta distribution for the chosen arm
        if reward == 1.0:
            self.alpha[arm] += 1
        elif reward == 0.0:
            self.beta[arm] += 1
        else:
            raise ValueError("Reward must be 0.0 or 1.0")

class GP_Bandit(MultiArmedBanditAlgorithm):

    """
    The GP-Bandit method is implemented using the Ax library.
    """
    def __init__(self, n_arms: int, batch_size: int = 1000):
        super().__init__(n_arms)
        self.client = Client()
        # Ax only allows for inequelity constraints. To force equality (all weights sum to 1) I do:
        # w_A + w_B + w_C <= 1
        # w_D = 1 - (w_A + w_B + w_C)
        # w_A >= 0, w_B >= 0, w_C >= 0
        # So, in case of n_arms, I need to create n_arms - 1 variables.

        # Create n - 1 variables
        parameters = [RangeParameterConfig(name=f"w_{i}", bounds=(0.0, 1.0), parameter_type="float") for i in range(n_arms - 1)]
        parameters_constraints = [
            # The sum of the variables must be less than or equal to 1
            " + ".join([f"w_{i}" for i in range(n_arms - 1)]) + " <= 1.0"
        ]
        self.client.configure_experiment(name="GP_Bandit", parameters=parameters, parameter_constraints=parameters_constraints)
        self.client.configure_optimization(objective="clicks")

        # Get initial set of weights
        trials = self.client.get_next_trials(max_trials=1)
        trial_index, parameters = list(trials.items())[0]
        self.weights = [parameters[f"w_{i}"] for i in range(n_arms - 1)]
        # This is a trick to ensure the weights sum to 1 in Ax
        last_weight = 1 - sum(self.weights)
        # Some times rounding errors make the last weight slightly negative, so we set it to 0
        if last_weight <= 0:
            last_weight = 0
        self.weights.append(last_weight)
        self.last_trial_index = trial_index
        self.batch_index = 0
        self.batch_size = batch_size
        self.batch_reward = 0

    def select_arm(self) -> int:
        # Select the arm based on the current weights
        return np.random.choice(len(self.weights), p=self.weights)

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        self.batch_reward += reward
        self.batch_index += 1
        if self.batch_index == self.batch_size:
            # After a number of experiments, we call Ax to update the GP model and get a new set of weights
            self.client.complete_trial(trial_index=self.last_trial_index, raw_data={"clicks": self.batch_reward})
            self.batch_reward = 0
            self.batch_index = 0
            # Get new set of weights
            trials = self.client.get_next_trials(max_trials=1)
            trial_index, parameters = list(trials.items())[0]
            self.weights = [parameters[f"w_{i}"] for i in range(self.n_arms - 1)]
            last_weight = 1 - sum(self.weights)
            # Some times rounding errors make the last weight slightly negative, so we set it to 0
            if last_weight <= 0:
                last_weight = 0
            self.weights.append(last_weight)
            self.last_trial_index = trial_index

class SimulationExperiment:
    def __init__(self, ctr_means: list[float], ctr_stds: list[float], algorithm: MultiArmedBanditAlgorithm):
        if len(ctr_means) != len(ctr_stds):
            raise ValueError("ctr_means and ctr_stds must have the same length")
        if len(ctr_means) != algorithm.n_arms:
            raise ValueError("ctr_means and ctr_stds must have the same length as the number of arms")

        self.ctr_means = ctr_means
        self.ctr_stds = ctr_stds
        self.algorithm = algorithm

    def run_experiment(self, n_trials: int) -> tuple[float, int, np.ndarray, np.ndarray]:
        # Stats to calculate the results of the experiment
        # How many clicks were accumulated over time
        accum_rewards_per_trial = np.zeros(n_trials)
        # How many times each arm was pulled
        pulls_per_arm_per_trial = np.zeros((n_trials, self.algorithm.n_arms))
        # Expected rewards for each arm (Will be used to calculate the regret)
        expected_rewards_per_arm = np.zeros(self.algorithm.n_arms)
        for i in range(n_trials):
            arm = self.algorithm.select_arm()
            ctr_mean = self.ctr_means[arm]
            ctr_std = self.ctr_stds[arm]
            ctr = np.clip(np.random.normal(ctr_mean, ctr_std), 0, 1)
            reward = np.random.binomial(1, ctr)
            self.algorithm.update(arm, reward)

            # For the selected arm, update the reward
            accum_rewards_per_trial[i] = accum_rewards_per_trial[i-1] + reward if i > 0 else reward
            # For the selected arm, update the number of pulls
            pulls_per_arm_per_trial[i, arm] = (pulls_per_arm_per_trial[i-1, arm] + 1) if i > 0 else 1

            # The expected value is updated on the selected arm
            expected_rewards_per_arm[arm] = (expected_rewards_per_arm[arm] * pulls_per_arm_per_trial[i, arm] + reward) / (pulls_per_arm_per_trial[i, arm] + 1)
            # The arms that were not selected are updated with the same value as the previous trial
            for j in range(self.algorithm.n_arms):
                if j != arm:
                    pulls_per_arm_per_trial[i, j] = pulls_per_arm_per_trial[i-1, j]
                    expected_rewards_per_arm[j] = expected_rewards_per_arm[j]
            
        # Calculate the regret for each trial
        # Now that the experiment is over, we have the final expected reward for each arm and we can calculate the total reward if we chose the best arm all the time.
        accumulated_regret_per_trial = np.zeros(n_trials)
        for j in range(n_trials):
            accumulated_regret_per_trial[j] = max(expected_rewards_per_arm) * (j+1) - accum_rewards_per_trial[j]
        
        return self.algorithm.total_reward, self.algorithm.total_trials, accum_rewards_per_trial, pulls_per_arm_per_trial, expected_rewards_per_arm, accumulated_regret_per_trial