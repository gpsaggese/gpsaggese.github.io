# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import msml610_utils as ut

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %%
import L09_05_01_discrete_bayes_dog_utils as ut

# %%
# !sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet filterpy)"

# %% [markdown]
# # Tracking a Dog

# %% [markdown]
# ## Problem Definition

# %% [markdown]
# - There is a dog with a sensor
# - The dog wanders around the offices and halls
# - The sensor reports if the dog is in front of a door or a hall
# - There are 10 positions in the hallway, numbered 0 to 9
# - The hallway is circular so after 9, the position is 0

# %% [markdown]
# ## A Simple Example

# %%
# At the beginning, we don't know where the dog is.
# The prior is: all the positions are equiprobable.
belief = np.array([1/10]*10)
print(belief)

# %%
ut.plot_belief(belief)

# %%
# The map of the office is the following.
hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
ut.plot_belief(hallway)

# %%
- The sensor returns always the correct answer.
- The first reading from the sensor is "door"
- The dog is in front of a door, but we don't know which one

# %%
belief = np.array([1/3, 1/3, 0, 0, 0, 0, 0, 0, 1/3, 0])
ut.plot_belief(belief)

# %% [markdown]
# - The readings is "door", "move right", "door"
# - The only location possible is position #1

# %%
belief = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
ut.plot_belief(belief)

# %% [markdown]
# ## Noisy Sensors

# %% [markdown]
# - If the sensor is not reliable, it seemed that it's impossible to determine where the dog is
# - How can you conclude anything, if you are always unsure?
#

# %%
belief = np.array([.31, .31, .01, .01, .01, .01, .01, .01, .31, .01])
ut.plot_belief(belief)

# %%
- Testing shows that the sensor is 3 times more likely to be right than wrong


# %%
def update_belief(
    hall: np.ndarray, belief: np.ndarray, z: int, correct_scale: float
) -> None:
    """
    Update belief in-place based on a measurement.

    Scales belief values by correct_scale for positions that match the
    measurement z.

    :param hall: Array representing the hallway map (0=wall, 1=door)
    :param belief: Array representing current belief distribution
    :param z: Measurement value (0 or 1)
    :param correct_scale: Scale factor for matching positions
    """
    for i, val in enumerate(hall):
        if val == z:
            belief[i] *= correct_scale

belief = np.array([0.1] * 10)
reading = 1 # 1 is 'door'
update_belief(hallway, belief, z=reading, correct_scale=3.)
print('belief:', belief)
print('sum =', sum(belief))
belief /= sum(belief)
ut.plot_belief(belief)

# %%
from filterpy.discrete_bayes import normalize

def scaled_update(
    hall: np.ndarray, belief: np.ndarray, z: int, z_prob: float
) -> None:
    """
    Update belief using scaled likelihood based on measurement probability.

    Computes a scale factor from the measurement probability and applies it
    to positions matching the measurement, then normalizes.

    :param hall: Array representing the hallway map (0=wall, 1=door)
    :param belief: Array representing current belief distribution
    :param z: Measurement value (0 or 1)
    :param z_prob: Probability that the measurement is correct
    """
    scale = z_prob / (1. - z_prob)
    belief[hall==z] *= scale
    normalize(belief)

belief = np.array([0.1] * 10)
scaled_update(hallway, belief, z=1, z_prob=.75)

print('sum =', sum(belief))
print('probability of door =', belief[0])
print('probability of wall =', belief[2])
ut.plot_belief(belief)

# %% [markdown]
# - Generalizing;
#   posterior = likelihood * prior / normalization

# %%
from filterpy.discrete_bayes import update

def lh_hallway(hall: np.ndarray, z: int, z_prob: float) -> np.ndarray:
    """
    Compute likelihood that a measurement matches positions in the hallway.

    Creates a likelihood array where positions matching the measurement z
    are scaled according to the measurement probability.

    :param hall: Array representing the hallway map (0=wall, 1=door)
    :param z: Measurement value (0 or 1)
    :param z_prob: Probability that the measurement is correct
    :return: Likelihood array for all positions
    """
    try:
        scale = z_prob / (1. - z_prob)
    except ZeroDivisionError:
        scale = 1e8
    likelihood = np.ones(len(hall))
    likelihood[hall==z] *= scale
    return likelihood

belief = np.array([0.1] * 10)
likelihood = lh_hallway(hallway, z=1, z_prob=.75)
# def update(likelihood, prior):
#    return normalize(likelihood * prior)
update(likelihood, belief)


# %% [markdown]
# ## Incorporating movement
#
# - Assume that the movement sensor is perfect
# - If the dog has moved to the right, we need to shift the belief to the right

# %%
def perfect_predict(belief: np.ndarray, move: int) -> np.ndarray:
    """
    Move the position by `move` spaces with perfect prediction.

    Shifts the belief distribution where positive is to the right, and
    negative is to the left. Uses circular indexing for wrap-around.

    :param belief: Array representing current belief distribution
    :param move: Number of positions to move (positive=right, negative=left)
    :return: Updated belief distribution after movement
    """
    n = len(belief)
    result = np.zeros(n)
    for i in range(n):
        result[i] = belief[(i-move) % n]
    return result


belief = np.array([.35, .1, .2, .3, 0, 0, 0, 0, 0, .05])
ut.plot_belief(belief)

# %%
new_belief = perfect_predict(belief, 1)
ut.plot_belief(new_belief)


# %% [markdown]
# ## Terminology
#
# - system: what we are trying to model
#     - E.g., the dog
# - state: configuration of the system
#     - E.g., the position of the dog
# - The filter produces an estimated state of the system
# - process model: the dog moves one or more positions at each time st4ep

# %% [markdown]
# ## Adding Uncertainty to the Prediction
#
# - Assume that the sensor's movement measurement $z$ is
#   - 80% to be correct
#   - 10% to overshoot by 1
#   - 10% to undershoot by 1
#
# - If movement measurement is 4, then the dog is:
#   - 80% likely to have moved to the right
#   - 10% likely to have moved 3 or 5 spaces to the right

# %%
def predict_move(
    belief: np.ndarray,
    move: int,
    p_under: float,
    p_correct: float,
    p_over: float,
) -> np.ndarray:
    """
    Predict movement with uncertainty in the motion model.

    Models imperfect movement where the actual displacement can differ from
    the measured movement by ±1 position with specified probabilities.

    :param belief: Array representing current belief distribution
    :param move: Measured movement (number of positions)
    :param p_under: Probability of undershooting by 1 position
    :param p_correct: Probability of correct movement
    :param p_over: Probability of overshooting by 1 position
    :return: Prior belief distribution after movement prediction
    """
    n = len(belief)
    prior = np.zeros(n)
    for i in range(n):
        prior[i] = (
            belief[(i-move) % n]   * p_correct +
            belief[(i-move-1) % n] * p_over +
            belief[(i-move+1) % n] * p_under)
    return prior

belief = [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
ut.plot_belief(belief)

move = 2
prior = predict_move(belief, move, .1, .8, .1)
ut.plot_belief(prior)

# %%
# Assume the belief is not correct.
belief = [0, 0, .4, .6, 0, 0, 0, 0, 0, 0]
ut.plot_belief(belief)

move = 2
prior = predict_move(belief, move, .1, .8, .1)
ut.plot_belief(prior)

# %% [markdown]
# - After the update with the noisy sensor there is always some lost information

# %%
from ipywidgets import interact, IntSlider

# %%
belief = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
predict_beliefs = []
predict_beliefs.append(belief)
print('Initial belief:', belief)

for i in range(100):
    belief = predict_move(belief, 1, .1, .8, .1)
    predict_beliefs.append(belief)
print('Final Belief:', belief)

# Make an interactive plot.
def show_prior(step: int) -> None:
    """
    Display belief distribution at a specific step.

    Interactive callback function to visualize how belief evolves over time.

    :param step: Time step to display (1-indexed)
    """
    ut.plot_belief(predict_beliefs[step-1])
    plt.title(f'Step {step}')
    plt.show()

interact(show_prior, step=IntSlider(value=1, max=len(predict_beliefs)));

# %%
