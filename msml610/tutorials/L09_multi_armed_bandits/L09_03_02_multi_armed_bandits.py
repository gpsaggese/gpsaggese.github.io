# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Multi-armed bandits

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style.
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %%
import helpers.hintrospection as hintros
import helpers.htutorial as ut
import L09_03_multi_armed_bandits_sim as sim
import L09_03_multi_armed_bandits_utils as utils

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# # Part 1: Introduction: Casino Slot Machines

# %% [markdown]
# ## Cell 1.1: Playing the casino
#
# **Goal**:
# - There are 3 slot machines, and you have 10 coins
# - Each machine gives you a payout in $[-1, 1]$ with an unknown mean
#   $\mu_i$
# - Choose which machine to play, and track total winnings and coin
#   budget
# - How do you maximize your winnings?
#
# **Implementation**: `utils.cell1_casino_slot_machines()`
# - Each machine is drawn as a placeholder box showing its last reward,
#   sample mean, and pull count once played

# %%
hintros.print_obj_info(utils.cell1_casino_slot_machines)

# %%
utils.cell1_casino_slot_machines()

# %% [markdown]
# **Usage**
# - Inputs
#   - **`random seed`**: seed for the hidden machine rewards
#   - **`number of coins`**: total coins available to play
#   - **`Show True Means`**: reveal the hidden $\mu_i$ for each machine
#   - **`Play Machine 1/2/3`**: spend one coin pulling that machine
#   - **`Reset Game`**: start over with a fresh coin budget
#
# - Panels
#   - **`Machine 1/2/3`**: each machine's last reward, sample mean, and
#     pull count so far

# %% [markdown]
# **Guided usage**
# - Play each machine once, then keep playing the one with the best
#   sample mean so far
#   - Observe the sample means are noisy early on and can be misleading
# - Toggle `Show True Means`
#   - Observe how far the sample means still are from the hidden $\mu_i$
#     after only a few pulls

# %% [markdown]
# ## Core Classes

# %% [markdown]
# The interactive cells above and below are built on top of a few core
# classes:
#
# | Object | Description | Comments |
# |--------|-------------|----------|
# | `MultiArmedBandit` | Environment with $K$ machines, each with a fixed but unknown true mean $\mu_i$ | Tracks pulls and rewards per machine |
# | `Strategy` | Abstract base class for a machine-selection policy | Subclasses: `ExplorationStrategy`, `ExploitationStrategy`, `EpsilonGreedyStrategy` |
# | `BanditExperiment` | Runs one `MultiArmedBandit` with one `Strategy` for $N$ coins | Returns rewards and cumulative rewards |
# | `BanditSimulation` | Runs many `BanditExperiment` trials with different seeds | Aggregates mean/std statistics across trials |

# %%
# Show the public API and GitHub source link of the core classes.
for cls in [
    sim.MultiArmedBandit,
    sim.Strategy,
    sim.BanditExperiment,
    sim.BanditSimulation,
]:
    hintros.print_obj_info(cls)

# %% [markdown]
# # Part 2: Exploration vs Exploitation Dilemma

# %% [markdown]
# ## Cell 2.1: Comparing the 3 basic strategies
#
# **Goal**:
# - Demonstrate the fundamental tradeoff between exploration and
#   exploitation on the same 3 slot machines, each with a fixed but
#   unknown true mean $\mu_i$
# - Instead of playing one coin at a time, play $N$ coins automatically,
#   and compare the total reward earned by each strategy
#
# **Strategies**:
# - **Pure exploration**: pick a machine uniformly at random on every
#   coin. Learns the true means accurately but wastes coins on bad
#   machines
# - **Pure exploitation**: pull each machine once, then always pick the
#   machine with the highest observed mean. Can get stuck on a
#   suboptimal machine if an early random reward looks good
# - **Balanced (epsilon-greedy)**: explore with probability $\epsilon$,
#   otherwise exploit the best known machine. Balances the 2 extremes:
#   $\epsilon$ controls how much exploration is kept
#
# **Implementation**: `utils.cell2_exploration_vs_exploitation()`

# %%
hintros.print_obj_info(utils.cell2_exploration_vs_exploitation)

# %%
utils.cell2_exploration_vs_exploitation()

# %% [markdown]
# **Usage**
# - Inputs
#   - **`random seed`**: seed for the hidden machine rewards
#   - **`number of coins`**: number of coins each strategy plays
#   - **`exploration probability`**: $\epsilon$ for the balanced strategy
#
# - Panels
#   - **`left`**: cumulative reward over time for each of the 3
#     strategies

# %% [markdown]
# **Guided usage**
# - Compare the 3 strategies at the default settings
#   - Observe pure exploration learns but earns little, pure
#     exploitation gets stuck on suboptimal choices, and a balance is
#     key
# - Lower `exploration probability` toward 0
#   - Observe the balanced strategy's curve approaches pure exploitation

# %% [markdown]
# # Part 3: Greedy Algorithm Failure

# %% [markdown]
# ## Cell 3.1: Watching greedy get stuck
#
# **Goal**:
# - See the greedy algorithm get permanently stuck on a suboptimal arm
# - Understand why pure exploitation is not enough
#
# **Implementation**: `utils.cell3_greedy_algorithm_failure()`

# %%
hintros.print_obj_info(utils.cell3_greedy_algorithm_failure)

# %%
utils.cell3_greedy_algorithm_failure()

# %% [markdown]
# **Usage**
# - Inputs
#   - **`random seed`**: seed for the hidden machine rewards and pulls
#   - **`number of coins`**: number of coins to play
#   - **`Run Greedy Algorithm`**: replay with the current settings
#
# - Panels
#   - **`Pull timeline`**: which machine was pulled at each round and the
#     reward it returned, color-coded by machine
#   - **`Empirical mean estimates`**: empirical mean of each machine over
#     time, with the (usually hidden) true means shown as dotted lines
#   - **`Comments`**: current seed, pull counts, and whether greedy got
#     stuck

# %% [markdown]
# **Guided usage**
# - Run with the default seed
#   - Observe greedy pulls each machine once, then always exploits the
#     best one so far
#   - If the first pull of a suboptimal machine returns a lucky high
#     reward, greedy locks onto it and never revisits the truly best
#     machine again: this is why greedy alone has linear regret, it
#     never corrects an early mistake
# - Try different seeds
#   - Observe how often greedy gets stuck vs finds the best machine by
#     luck

# %% [markdown]
# # Part 4: Epsilon-Greedy Algorithm

# %% [markdown]
# ## Cell 4.1: Fixing greedy with a little exploration
#
# **Goal**:
# - See how a small exploration probability $\epsilon$ prevents the
#   "stuck forever" failure of pure greedy from Part 3
#
# **Implementation**: `utils.cell4_epsilon_greedy()`

# %%
hintros.print_obj_info(utils.cell4_epsilon_greedy)

# %%
utils.cell4_epsilon_greedy()

# %% [markdown]
# **Usage**
# - Inputs
#   - **`random seed`**: seed for the hidden machine rewards and pulls
#   - **`number of coins`**: number of coins to play
#   - **`exploration probability`**: $\epsilon$
#   - **`Run Epsilon-Greedy`**: replay with the current settings
#
# - Panels
#   - **`Pull timeline`**: pulls color-coded by decision type
#     (gray=init, blue=explore, green=exploit)
#   - **`Pull counts`**: number of times each machine was pulled
#   - **`Cumulative reward`**: total reward earned over time
#   - **`Comments`**: current seed, epsilon, decision counts, pull
#     counts

# %% [markdown]
# **Guided usage**
# - Set `exploration probability` to `0.1`
#   - Observe about 10% of rounds explore (random machine) and 90%
#     exploit (best known machine); occasional exploration lets
#     epsilon-greedy discover and recover from an early unlucky
#     estimate, unlike pure greedy
# - Set `exploration probability` to `0`
#   - Observe it recovers the greedy algorithm from Part 3
# - Increase `exploration probability`
#   - Observe the machine pulled uniformly more often, at the cost of
#     exploiting less

# %% [markdown]
# # Part 5: Confidence Intervals for Each Arm

# %% [markdown]
# ## Cell 5.1: Quantifying uncertainty per arm
#
# **Goal**:
# - Introduce confidence bounds and uncertainty quantification for the
#   empirical mean of each arm
#
# **Implementation**: `utils.cell5_confidence_intervals()`

# %%
hintros.print_obj_info(utils.cell5_confidence_intervals)

# %%
utils.cell5_confidence_intervals()

# %% [markdown]
# **Usage**
# - Inputs
#   - **`random seed`**: seed for the pulls
#   - **`pulls per machine`**: number of pulls $N$ per machine
#   - **`confidence`**: confidence level (e.g., 90%, 95%, 99%)
#   - **`Show True Means`**: reveal the hidden $\mu_i$ as dotted lines
#
# - Panels
#   - **`Empirical mean with CI`**: bar chart of empirical mean per
#     machine with a Hoeffding confidence-interval error bar
#   - **`CI half-width vs N`**: theoretical curve of how the half-width
#     shrinks as the number of pulls grows, with a marker at the current
#     $N$
#   - **`Comments`**: current seed, $N$, confidence level, and numeric CI
#     bounds

# %% [markdown]
# **Guided usage**
# - Raise `pulls per machine`
#   - Observe the confidence interval shrinks: uncertainty about $\mu_i$
#     decreases as $1/\sqrt{N}$
# - Raise `confidence` from 90% to 99%
#   - Observe the interval widens, since it must hold with higher
#     probability
# - Toggle `Show True Means`
#   - Observe whether the true mean actually falls inside the interval;
#     this shrinking uncertainty radius is exactly the "exploration
#     bonus" used by UCB in Part 6

# %% [markdown]
# # Part 6: Upper Confidence Bound (UCB) Intuition

# %% [markdown]
# ## Cell 6.1: UCB as mean plus a bonus
#
# **Goal**:
# - See the UCB index as empirical mean plus an exploration bonus
# - Build intuition for why UCB can prefer a machine with a lower
#   empirical mean if it has been pulled fewer times
#
# **Implementation**: `utils.cell6_ucb_intuition()`

# %%
hintros.print_obj_info(utils.cell6_ucb_intuition)

# %%
utils.cell6_ucb_intuition()

# %% [markdown]
# **Usage**
# - Inputs
#   - **`current round`**: $t$, the round used to compute every bonus
#
# - Panels
#   - **`UCB index`**: stacked bar chart with empirical mean (blue)
#     stacked with the exploration bonus (orange); the machine with the
#     highest total (marked `*`) would be pulled next
#   - **`Comments`**: current $t$, and each machine's $N_i$, mean,
#     bonus, and UCB index

# %% [markdown]
# **Guided usage**
# - Read the machine with the lowest empirical mean
#   - Observe its small $N_i$ gives it a large bonus, which can make its
#     UCB the highest: this is "optimism in the face of uncertainty",
#     under-explored arms get the benefit of the doubt
# - Increase `current round`
#   - Observe every machine's bonus grows a little, since $\log t$ grows
#     for all arms regardless of which one is pulled

# %% [markdown]
# # Part 7: UCB Algorithm Simulation

# %% [markdown]
# ## Cell 7.1: Running UCB1 end-to-end
#
# **Goal**:
# - Watch UCB1 run end-to-end on 4 arms and converge to the best one
#
# **Implementation**: `utils.cell7_ucb_simulation()`

# %%
hintros.print_obj_info(utils.cell7_ucb_simulation)

# %%
utils.cell7_ucb_simulation()

# %% [markdown]
# **Usage**
# - Inputs
#   - **`random seed`**: seed for the hidden means and pulls
#   - **`time horizon`**: $T$, number of rounds to simulate
#   - **`Run UCB Algorithm`**: replay with the current settings
#
# - Panels
#   - **`Pull timeline`**: which machine was pulled at each round
#   - **`Pull counts over time`**: $N_i(t)$ for each machine as the run
#     progresses
#   - **`Cumulative regret`**: $L_t$ over time
#   - **`Comments`**: pull counts, optimal arm, and final regret

# %% [markdown]
# **Guided usage**
# - Run with the default settings
#   - Observe UCB1 quickly identifies Machine 3 (the true best arm,
#     $\mu=0.7$) and pulls it most often, while occasionally revisiting
#     the others
#   - Observe the cumulative regret curve flattens over time: its slope
#     decreases as $\log t$ grows more slowly than $t$
# - Increase `time horizon`
#   - Observe the pull counts on the suboptimal arms grow much more
#     slowly than the optimal arm's

# %% [markdown]
# # Part 8: UCB Exploration Bonus Decay

# %% [markdown]
# ## Cell 8.1: Isolating the bonus term
#
# **Goal**:
# - Isolate how the UCB exploration bonus $\sqrt{2 \log(t) / N_i}$
#   depends on $N_i$ alone
#
# **Implementation**: `utils.cell8_ucb_bonus_decay()`

# %%
hintros.print_obj_info(utils.cell8_ucb_bonus_decay)

# %%
utils.cell8_ucb_bonus_decay()

# %% [markdown]
# **Usage**
# - Inputs
#   - **`current round`**: $t$, fixed while `pulls of arm i` varies
#   - **`pulls of arm i`**: $N_i$, the number of pulls of the arm under
#     inspection
#
# - Panels
#   - **`Bonus vs N_i`**: curve of the exploration bonus as a function
#     of the number of pulls, at a fixed round $t$, with a marker at the
#     current $N_i$
#   - **`Comments`**: current $t$, $N_i$, and the resulting bonus value

# %% [markdown]
# **Guided usage**
# - Double `pulls of arm i`
#   - Observe the bonus does not halve: it shrinks by a factor of
#     $\sqrt{2}$, since the bonus decays as $1/\sqrt{N_i}$
# - Increase `current round`
#   - Observe the whole curve shifts up slightly (via $\sqrt{\log t}$),
#     but $N_i$ dominates the shape of the decay

# %% [markdown]
# # Part 9: Regret Accumulation

# %% [markdown]
# ## Cell 9.1: Per-step vs cumulative regret
#
# **Goal**:
# - Visualize how per-step and cumulative regret accumulate for a chosen
#   algorithm
#
# **Implementation**: `utils.cell9_regret_accumulation()`

# %%
hintros.print_obj_info(utils.cell9_regret_accumulation)

# %%
utils.cell9_regret_accumulation()

# %% [markdown]
# **Usage**
# - Inputs
#   - **`algorithm`**: Random, Greedy, Epsilon-Greedy, or UCB
#   - **`random seed`**: seed for the hidden means and pulls
#   - **`time horizon`**: $T$, number of rounds to simulate
#   - **`Run Algorithm`**: replay with the current settings
#
# - Panels
#   - **`Per-step regret`**: bar chart of instantaneous regret
#     $\ell_t = \mu^* - \mu_{A_t}$ at every round, colored green when
#     the optimal arm was chosen
#   - **`Cumulative regret`**: line plot of
#     $L_t = \sum_{\tau \le t} \ell_\tau$
#   - **`Comments`**: algorithm, optimal-arm pull count, and final
#     regret

# %% [markdown]
# **Guided usage**
# - Select `Random` or `Greedy`
#   - Observe regret accumulates steadily at every round (bars rarely
#     turn green)
# - Select `Epsilon-Greedy` or `UCB`
#   - Observe mostly green bars once the algorithm locks onto the best
#     arm, with occasional red bars from exploration
# - Switch the `algorithm` dropdown across all 4 options
#   - Compare how differently each one's cumulative regret curve bends

# %% [markdown]
# # Part 10: Comparing Algorithms: Regret Curves

# %% [markdown]
# ## Cell 10.1: Regret growth rate across algorithms
#
# **Goal**:
# - Compare the regret growth rate of Random, Greedy, Epsilon-Greedy,
#   UCB, and Thompson Sampling on the same log-t axis
#
# **Implementation**: `utils.cell10_regret_comparison()`

# %%
hintros.print_obj_info(utils.cell10_regret_comparison)

# %%
utils.cell10_regret_comparison()

# %% [markdown]
# **Usage**
# - Inputs
#   - **`base random seed`**: seed shared across the trials being
#     averaged
#   - **`number of arms`**: $K$
#   - **`T (time horizon)`**: given as $\log_2(T)$
#   - **`Run Comparison`**: replay with the current settings
#
# - Panels
#   - **`Regret curves`**: mean cumulative regret (averaged over a few
#     trials) for each selected algorithm, log scale on the round axis
#   - **`Comments`**: final regret for each selected algorithm, with its
#     theoretical growth rate

# %% [markdown]
# **Guided usage**
# - Compare Random/Greedy against UCB/Thompson Sampling
#   - Observe Random and Greedy grow linearly in $T$ ($\Theta(T)$):
#     their curves keep climbing even on a log-t axis
#   - Observe Epsilon-Greedy (fixed $\epsilon$) also grows roughly
#     linearly, since it never stops exploring
#   - Observe UCB and Thompson Sampling flatten out on the log-t axis,
#     consistent with $O(\log T)$ regret
# - Increase `number of arms`
#   - Observe all algorithms get worse, but UCB and Thompson Sampling
#     degrade much more gracefully

# %% [markdown]
# # Part 11: Bayesian Bandits: Prior and Posterior

# %% [markdown]
# ## Cell 11.1: Updating a belief with data
#
# **Goal**:
# - Introduce Bayesian inference for bandits: start from a prior belief
#   and update it with observed data
#
# **Implementation**: `utils.cell11_bayesian_prior_posterior()`

# %%
hintros.print_obj_info(utils.cell11_bayesian_prior_posterior)

# %%
utils.cell11_bayesian_prior_posterior()

# %% [markdown]
# **Usage**
# - Inputs
#   - **`random seed`**: seed for the simulated pulls
#   - **`prior alpha`**, **`prior beta`**: the Beta prior's parameters
#   - **`Pull Arm`**: simulate one more pull and update the posterior
#   - **`Reset`**: clear pulls and return to the prior
#
# - Panels
#   - **`Prior vs posterior`**: prior $\text{Beta}(\alpha, \beta)$
#     (dotted) and posterior $\text{Beta}(\alpha+s, \beta+f)$ (solid,
#     shaded) probability density over the unknown success probability
#     $\mu$
#   - **`Comments`**: successes $s$, failures $f$, and posterior
#     mean/variance

# %% [markdown]
# **Guided usage**
# - Start from a flat prior $\text{Beta}(1,1)$ and click `Pull Arm`
#   repeatedly
#   - Observe each pull narrows the posterior around the true (hidden)
#     success probability, and the posterior mean moves toward the
#     observed success rate while its variance shrinks
# - Set a more concentrated prior (large `prior alpha`, `prior beta`)
#   - Observe it takes more pulls to move the posterior away from the
#     prior belief

# %% [markdown]
# # Part 12: Thompson Sampling Algorithm

# %% [markdown]
# ## Cell 12.1: Sampling from each arm's posterior
#
# **Goal**:
# - See Thompson Sampling sample one $\theta_i$ from each arm's
#   posterior and pull the arm with the highest sample
#
# **Implementation**: `utils.cell12_thompson_sampling()`

# %%
hintros.print_obj_info(utils.cell12_thompson_sampling)

# %%
utils.cell12_thompson_sampling()

# %% [markdown]
# **Usage**
# - Inputs
#   - **`random seed`**: seed for the hidden means and pulls
#   - **`number of arms`**, **`number of rounds`**: size of the
#     simulated run
#   - **`round to inspect`**: which round's posteriors and samples to
#     display
#   - **`Run Thompson Sampling`**: replay with the current settings
#
# - Panels
#   - **`Posterior curves`**: one Beta posterior density per arm at the
#     chosen round, with a dot at each arm's sampled $\theta_i$ (a star
#     marks the arm that was actually pulled)
#   - **`Comments`**: each arm's posterior parameters, sampled value,
#     and the round's selection

# %% [markdown]
# **Guided usage**
# - Set `round to inspect` to an early round
#   - Observe wide posteriors mean any arm's sample can win, so
#     exploration is automatic; the selected arm is always the one
#     whose sampled $\theta_i$ happens to be highest that round, not
#     necessarily the one with the highest posterior mean
# - Scrub `round to inspect` forward
#   - Observe the posteriors narrow and the sampled dots cluster
#     increasingly around the true best arm

# %% [markdown]
# # Part 13: Thompson Sampling: Probability Matching

# %% [markdown]
# ## Cell 13.1: Verifying probability matching
#
# **Goal**:
# - Verify that Thompson Sampling selects each arm with probability
#   exactly equal to the probability that arm is optimal given the data
#
# **Implementation**: `utils.cell13_probability_matching()`

# %%
hintros.print_obj_info(utils.cell13_probability_matching)

# %%
utils.cell13_probability_matching()

# %% [markdown]
# **Usage**
# - Inputs
#   - **`random seed`**: seed for the simulated pulls
#   - **`pulls per arm (data)`**: how much data each arm's posterior is
#     conditioned on
#   - **`Run 1000 Steps`**: resample 1000 more Thompson Sampling draws
#     from the fixed posterior
#
# - Panels
#   - **`Theoretical Pr(optimal)`**: bar chart of
#     $\Pr(i = i^* \mid \mathcal{D})$ for each arm, estimated with a
#     large number of posterior draws
#   - **`Empirical frequency`**: bar chart of how often each arm wins
#     when sampling 1000 more times from the same fixed posterior
#   - **`Comments`**: posterior parameters and the theoretical vs
#     empirical numbers

# %% [markdown]
# **Guided usage**
# - Compare the 2 bar charts at the default settings
#   - Observe they closely match: this is "probability matching", the
#     defining property of Thompson Sampling
#   - Observe an arm that is rarely optimal is rarely selected, but
#     never with exactly zero probability, so it can still be revisited
#     if new data supports it
# - Raise `pulls per arm (data)`
#   - Observe both bar charts concentrate on one machine, as more data
#     sharpens the posterior toward the truly best arm

# %% [markdown]
# # Part 14: UCB vs Thompson Sampling Comparison

# %% [markdown]
# ## Cell 14.1: Head-to-head on the same environment
#
# **Goal**:
# - Compare the 2 order-optimal algorithms empirically on the same
#   bandit environment
#
# **Implementation**: `utils.cell14_ucb_vs_thompson()`

# %%
hintros.print_obj_info(utils.cell14_ucb_vs_thompson)

# %%
utils.cell14_ucb_vs_thompson()

# %% [markdown]
# **Usage**
# - Inputs
#   - **`random seed`**: seed for the hidden means and pulls
#   - **`number of arms`**: $K$
#   - **`suboptimality gap`**: $\Delta$ between the best and next-best
#     arm
#   - **`time horizon`**: $T$
#   - **`Run Both Algorithms`**: replay with the current settings
#
# - Panels
#   - **`Regret curves`**: cumulative regret of UCB and Thompson
#     Sampling overlaid
#   - **`Pull counts`**: grouped bar chart of pull counts per arm for
#     each algorithm
#   - **`Comments`**: setup ($K$, $\Delta$, $T$) and each algorithm's
#     final regret

# %% [markdown]
# **Guided usage**
# - Run with the default settings
#   - Observe both algorithms achieve $O(\log T)$ regret, so their
#     cumulative regret curves both flatten out over time
# - Try several seeds
#   - Observe Thompson Sampling often (not always) ends with lower final
#     regret: it tends to have better constants in practice
# - Shrink `suboptimality gap`
#   - Observe both algorithms need more pulls of the suboptimal arms
#     before locking onto the best one, since the arms are harder to
#     distinguish
