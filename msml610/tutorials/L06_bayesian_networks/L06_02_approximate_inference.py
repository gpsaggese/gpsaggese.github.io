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
# # Approximate Inference in Bayesian Networks
#
# - This notebook teaches how to estimate posteriors $P(X \mid \mathbf{e})$ by
#   sampling, when exact inference is too expensive or impossible
# - Concepts are built on the Garden World examples with variables
#   to run the query $P(Rain \mid Sprinkler{=}T)$
# - The pedagogical arc:
#   - Turn uniform randomness into samples (inverse transform)
#   - Sample a whole network (prior sampling)
#   - Watch estimates converge ($1/\sqrt{N}$)
#   - Condition on evidence by rejection
#   - Rescue rare evidence with importance weights
#   - Walk the state space with MCMC (mixing, Gibbs, Metropolis-Hastings)
# - The exact posteriors from `pgmpy` are reused throughout as the ground-truth
#   reference that every estimate is compared against

# %%
# %load_ext autoreload
# %autoreload 2

import logging


# %%
import helpers.hintrospection as hintros
import helpers.hnotebook as hnotebook

import L06_02_approximate_inference_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# # Part 1: From Randomness to Samples

# %% [markdown]
# ## Cell 1.1: Turning uniform randomness into a discrete distribution
#
# **Goal**
# - Show that a stream of uniform numbers $r \in [0,1]$ becomes samples from
#   a discrete target, a biased die, via the inverse CDF

# %% [markdown]
# **Description**
# - Inputs
#   - `seed`: random seed for the uniform stream
#   - `N` (log scale): number of samples drawn
#
# - Panels
#   - `Target: biased die`: the die's probability mass function
#   - `CDF and inverse map`: the staircase CDF, with one sampled $r$
#     mapped to its face $x$
#   - `Sample histogram`: $N$ generated samples (solid) vs the target
#     (dotted)
#   - `Comments`: the construction and the achieved accuracy

# %%
# Map a uniform r through the die's CDF into a sampled face.
utils.cell1_1_inverse_transform_discrete_widget()

# %% [markdown]
# **Guided usage**
# - Raise `N` from small to large, leaving `seed` fixed
#   - Observe the sample histogram fill in toward the target PMF, and the
#     `max |freq - P|` error in Comments shrink
# - Change `seed` a few times at a small `N`
#   - Observe the single mapped point $r \to x$ jump to a different face
#     each time, while the underlying CDF never changes

# %% [markdown]
# **Implementation** `cell1_1_inverse_transform_discrete_widget()`
# - Draws `N` uniform numbers, then maps each through the die's inverse CDF
#   in `_plot_inverse_transform()`: the smallest face $x$ with $F(x) > r$
# - Overlays the resulting sample histogram on the target PMF

# %%
hintros.print_obj_info(utils.cell1_1_inverse_transform_discrete_widget)

# %% [markdown]
# ## Cell 1.2: Turning uniform randomness into a continuous distribution
#
# **Goal**
# - Show the same inverse-transform trick on a continuous target, an
#   exponential, where the inverse CDF has a closed form

# %% [markdown]
# **Description**
# - Inputs
#   - `seed`: random seed for the uniform stream
#   - `lambda`: rate of the exponential
#   - `N` (log scale): number of samples drawn
#
# - Panels
#   - `Target: exponential`: the exponential density
#   - `CDF and inverse map`: the smooth CDF, with one sampled $r$
#     mapped to its $x$
#   - `Sample histogram`: $N$ generated samples (solid) vs the target
#     density (dotted)
#   - `Comments`: the construction and the achieved accuracy

# %%
# Map a uniform r through the exponential's closed-form inverse CDF.
utils.cell1_2_inverse_transform_continuous_widget()

# %% [markdown]
# **Guided usage**
# - Raise `N` from small to large, leaving `lambda` fixed
#   - Observe the sample histogram fill in toward the target density, and
#     the sample mean in Comments approach the theoretical mean $1/\lambda$
# - Raise `lambda`
#   - Observe the target density and the histogram both compress toward 0,
#     since a larger rate means a smaller expected value
# - One trick underlies both cells: stretch a flat $[0,1]$ number through
#   the CDF and it comes out distributed like the target; when $F^{-1}$
#   has no closed form the same idea works with numerical inversion

# %% [markdown]
# **Implementation** `cell1_2_inverse_transform_continuous_widget()`
# - Draws `N` uniform numbers, then maps each through
#   $x = -\frac{1}{\lambda}\ln(1-r)$ in `_plot_inverse_transform()`, the
#   same helper Cell 1.1 uses
# - Overlays the resulting sample histogram on the target density

# %%
hintros.print_obj_info(utils.cell1_2_inverse_transform_continuous_widget)

# %% [markdown]
# ## Cell 1.3: Prior sampling from the sprinkler network
#
# **Goal**
# - Scale the single-variable trick up to a whole Bayesian network, by
#   sampling every variable in topological order to generate full events

# %% [markdown]
# **Description**
# - Inputs
#   - `seed`: random seed for the generated events
#   - `N` (log scale): number of full events to generate
#   - `Track marginal`: variable whose marginal estimate is tracked
#
# - Panels
#   - `Sample parents before children`: the sprinkler DAG, colored by
#     topological depth
#   - `Marginal of <var>`: estimated $P(\cdot)$ of the tracked variable
#     vs its exact value
#   - `Joint frequencies`: estimated joint over all 16 configurations
#     vs the exact joint
#   - `Comments`: the sampling order and the achieved errors

# %%
# Generate N complete worlds in topological order and compare with the exact joint.
utils.cell1_3_prior_sampling_widget()

# %% [markdown]
# **Guided usage**
# - Raise `N` from small to large, leaving `Track marginal` fixed
#   - Observe the estimated marginal bar converge onto the exact reference,
#     and the joint-frequency dots settle onto the exact joint curve
# - Switch `Track marginal` across all four variables
#   - Observe every one of them converges the same way: prior sampling
#     realizes the factorization $\prod_i \Pr(x_i \mid parents(X_i))$
#     regardless of which variable is tracked

# %% [markdown]
# **Implementation** `cell1_3_prior_sampling_widget()`
# - Generates `N` full events with `_prior_sample_array()`, sampling each
#   node only after its parents already have a value
# - Compares the estimated marginal of `Track marginal` and the estimated
#   joint over all 16 configurations against the exact `pgmpy` values

# %%
hintros.print_obj_info(utils.cell1_3_prior_sampling_widget)

# %% [markdown]
# ## Cell 1.4: Consistency and the 1/sqrt(N) convergence rate
#
# **Goal**
# - Make Monte Carlo convergence tangible, and show that error shrinks like
#   $1/\sqrt{N}$, setting expectations for every later sampler

# %% [markdown]
# **Description**
# - Inputs
#   - `seed`: base random seed for the independent chains
#   - `max N` (log scale): largest sample count shown
#   - `reps`: number of independent chains in the fan
#   - `Estimate`: which marginal event is being estimated
#
# - Panels
#   - `One estimate`: a single running estimate converging to the
#     exact value
#   - `<reps> independent chains`: the fan of chains narrowing as $N$
#     grows
#   - `Error vs N`: RMS error on log-log axes with a $-1/2$ reference
#     slope
#   - `Comments`: the exact value and the final error

# %%
# Show one estimate, many estimates, and the 1/sqrt(N) error rate.
utils.cell1_4_convergence_widget()

# %% [markdown]
# **Guided usage**
# - Watch the fan of chains as `max N` grows
#   - Observe every chain narrows toward the exact value, but never stops
#     wobbling: consistency guarantees convergence, not a fixed error at
#     any finite $N$
# - Compare the `Error vs N` curve against its $-1/2$ reference slope
#   - Observe the two track each other closely: 10x accuracy costs 100x
#     samples, which is why the rest of the notebook focuses on using each
#     sample better, not just drawing more of them

# %% [markdown]
# **Implementation** `cell1_4_convergence_widget()`
# - Runs `reps` independent running-estimate chains of the chosen `Estimate`
#   event up to `max N`, each from its own seed offset from `seed`
# - Plots the RMS error across chains against a $-1/2$ reference slope on
#   log-log axes

# %%
hintros.print_obj_info(utils.cell1_4_convergence_widget)

# %% [markdown]
# # Part 2: Conditioning on Evidence

# %% [markdown]
# ## Cell 2.1: Rejection sampling
#
# **Goal**
# - Introduce the simplest way to condition on evidence: generate prior
#   samples and throw away those that disagree with it

# %% [markdown]
# **Description**
# - Inputs
#   - `seed`: random seed for the prior samples
#   - `N` (log scale): total prior samples generated
#   - `Query X`: the query variable
#   - `observe <node>`: one checkbox plus True/False value per node,
#     marking it as evidence
#
# - Panels
#   - `Sample stream (first N)`: dots colored by kept (matches
#     evidence) vs rejected
#   - `Retained fraction`: counts of generated, rejected, and kept
#     samples
#   - `Posterior estimate`: $P(X \mid \mathbf{e})$ vs the exact
#     reference
#   - `Comments`: the retained fraction and the estimate

# %%
# Keep only the prior samples that agree with the evidence.
utils.cell2_1_rejection_sampling_widget()

# %% [markdown]
# **Guided usage**
# - Mark one more node as evidence, making it harder to satisfy
#   - Observe `Retained fraction`'s kept count drop sharply: the rarer the
#     evidence, the more samples are burned just to learn anything
# - Raise `N` at a fixed, fairly rare evidence setting
#   - Observe the posterior estimate keeps improving, but far slower than
#     `N` itself grows, since only the kept fraction actually contributes

# %% [markdown]
# **Implementation** `cell2_1_rejection_sampling_widget()`
# - Draws `N` prior samples and keeps only the ones matching every
#   `observe <node>` value
# - Estimates $P(X \mid \mathbf{e})$ from the kept samples, and reports the
#   retained fraction

# %%
hintros.print_obj_info(utils.cell2_1_rejection_sampling_widget)

# %% [markdown]
# ## Cell 2.2: Importance sampling and likelihood weighting
#
# **Goal**
# - Fix rejection's waste by keeping every sample and correcting with
#   importance weights instead of discarding disagreeing ones

# %% [markdown]
# **Description**
# - Inputs
#   - `seed`: random seed for the weighted samples
#   - `N` (log scale): number of weighted samples
#   - `Query X`: the query variable
#   - `observe <node>`: one checkbox plus True/False value per node,
#     marking it as evidence
#
# - Panels
#   - `Weighted samples (first N)`: dots sized by importance weight
#     $w = \Pr(X)/Q(X)$
#   - `Weight distribution`: histogram of weights, flagging collapse
#   - `Posterior estimate`: weighted estimate vs exact and vs
#     rejection
#   - `Comments`: the effective sample size and the estimates

# %%
# Keep every sample and correct the bias with importance weights.
utils.cell2_2_likelihood_weighting_widget()

# %% [markdown]
# **Guided usage**
# - Set the same rare evidence used in Cell 2.1, at the same `N`
#   - Observe the weight distribution grow uneven, but the posterior
#     estimate still tracks the exact reference: no sample was discarded
#     to get there
# - Compare the effective sample size in Comments against `N`
#   - Observe it fall well below `N` once weights get uneven: very
#     uneven weights shrink the effective sample size despite a large $N$

# %% [markdown]
# **Implementation** `cell2_2_likelihood_weighting_widget()`
# - Draws `N` samples with `_likelihood_weight_array()`, clamping evidence
#   nodes and weighting each sample by $w = \Pr(X)/Q(X)$
# - Reports the effective sample size alongside the weighted posterior
#   estimate

# %%
hintros.print_obj_info(utils.cell2_2_likelihood_weighting_widget)

# %% [markdown]
# # Part 3: Markov Chain Monte Carlo

# %% [markdown]
# ## Cell 3.1: Markov chains and the stationary distribution
#
# **Goal**
# - Introduce the core MCMC idea, a designed random walk whose long-run
#   distribution settles to a fixed shape independent of the start

# %% [markdown]
# **Description**
# - Inputs
#   - `seed`: random seed (unused by this deterministic walk, kept for
#     convention)
#   - `t`: number of steps taken
#   - `Initial state`: where the walk starts
#
# - Panels
#   - `Transition diagram`: states as nodes, with the current
#     most-likely state highlighted
#   - `State distribution`: $\pi_t$ (solid) settling onto the
#     stationary distribution (dotted)
#   - `Convergence`: total-variation distance to stationary decaying
#     with $t$
#   - `Comments`: the stationary distribution and current distance

# %%
# Step the chain and watch the state distribution converge to a fixed shape.
utils.cell3_1_markov_chain_widget()

# %% [markdown]
# **Guided usage**
# - Raise `t` from 0 toward 40, leaving `Initial state` fixed
#   - Observe $\pi_t$ settle onto the stationary distribution and the
#     total-variation distance decay toward 0
# - Change `Initial state` to a different state, then repeat
#   - Observe $\pi_t$ still converges to the exact same stationary shape:
#     the limit does not depend on where the walk starts

# %% [markdown]
# **Implementation** `cell3_1_markov_chain_widget()`
# - Steps a fixed transition matrix `t` times from `Initial state`, and
#   compares the evolving distribution $\pi_t$ against the exact
#   `_stationary_distribution()`
# - Tracks total-variation distance to stationary at every step

# %%
hintros.print_obj_info(utils.cell3_1_markov_chain_widget)

# %% [markdown]
# ## Cell 3.2: Mixing and burn-in
#
# **Goal**
# - Show that a correct stationary distribution is not enough: mixing
#   speed determines whether finite-sample estimates can be trusted

# %% [markdown]
# **Description**
# - Inputs
#   - `seed`: random seed for the chain
#   - `step`: proposal step size controlling mixing quality
#   - `burnin`: number of initial samples discarded
#   - `N` (log scale): total iterations
#
# - Panels
#   - `Trace (first N)`: the sampled value over iterations, with the
#     burn-in region shaded
#   - `Collected samples`: histogram of kept samples vs the true
#     bimodal posterior
#   - `Autocorrelation`: autocorrelation vs lag, high for poor mixing
#   - `Comments`: the mixing diagnostics

# %%
# Tune the step size between poor and good mixing and set the burn-in.
utils.cell3_2_mixing_burnin_widget()

# %% [markdown]
# **Guided usage**
# - Set `step` very small, then very large
#   - Observe the trace stay stuck near one mode when `step` is small, but
#     jump between both modes when `step` is large, with autocorrelation
#     dropping to match
# - Raise `burnin` at a small `step`
#   - Observe the histogram still misses one mode: a correct target and a
#     longer burn-in cannot fix a chain that never explores both modes

# %% [markdown]
# **Implementation** `cell3_2_mixing_burnin_widget()`
# - Runs a Metropolis chain on a bimodal target with `_metropolis_1d()`,
#   varying the proposal `step` size
# - Discards the first `burnin` samples, then compares the trace,
#   histogram, and autocorrelation of what remains

# %%
hintros.print_obj_info(utils.cell3_2_mixing_burnin_widget)

# %% [markdown]
# ## Cell 3.3: Gibbs sampling and the Markov blanket
#
# **Goal**
# - Specialize MCMC to Bayesian networks with Gibbs sampling: resample one
#   variable at a time from its Markov blanket, evidence held clamped

# %% [markdown]
# **Description**
# - Inputs
#   - `seed`: random seed for the sweeps
#   - `sweeps` (log scale): number of Gibbs sweeps
#   - `burnin`: burn-in sweeps discarded
#   - `Query X`: the query variable
#   - `observe <node>`: one checkbox plus True/False value per node,
#     marking it as evidence
#
# - Panels
#   - `Resample <var> from its blanket`: the DAG with evidence frozen
#     and the resampled variable plus its blanket highlighted
#   - `P(<var> | blanket)`: the full conditional currently being
#     sampled from
#   - `Running estimate`: $P(\text{Query X} \mid \mathbf{e})$ vs the
#     exact reference
#   - `Comments`: the hidden variables and the estimate

# %%
# Resample each hidden variable from its Markov blanket with evidence clamped.
utils.cell3_3_gibbs_sampling_widget()

# %% [markdown]
# **Guided usage**
# - Raise `sweeps` from small to large, leaving `burnin` fixed
#   - Observe the running estimate settle onto the exact reference
# - Watch which variable gets resampled across several sweeps
#   - Observe it cycles through only the hidden variables: evidence
#     variables stay clamped, so no rejection is ever needed

# %% [markdown]
# **Implementation** `cell3_3_gibbs_sampling_widget()`
# - Runs `sweeps` Gibbs sweeps with `_gibbs_chain()`, resampling each
#   hidden variable from `_gibbs_full_conditional()` while evidence nodes
#   stay fixed
# - Tracks the running posterior estimate for `Query X` after discarding
#   `burnin` sweeps

# %%
hintros.print_obj_info(utils.cell3_3_gibbs_sampling_widget)

# %% [markdown]
# ## Cell 3.4: Metropolis-Hastings and accept/reject moves
#
# **Goal**
# - Generalize beyond Gibbs to Metropolis-Hastings: correct an arbitrary
#   proposal with an acceptance probability instead of always accepting

# %% [markdown]
# **Description**
# - Inputs
#   - `seed`: random seed for the chain
#   - `p_local`: probability of a local single-variable move vs a
#     broad jump
#   - `iters` (log scale): number of iterations
#   - `burnin`: burn-in iterations discarded
#
# - Panels
#   - `Last proposed move`: current vs proposed state, with the
#     acceptance probability $A(x, x')$
#   - `Trace (accept rate)`: accepted states over iterations, with the
#     running acceptance rate
#   - `Running estimate`: $P(Rain \mid Sprinkler{=}T)$ vs the exact
#     reference
#   - `Comments`: the proposal mix, acceptance rate, and estimate

# %%
# Propose a move, then accept or reject it by the Hastings ratio.
utils.cell3_4_metropolis_hastings_widget()

# %% [markdown]
# **Guided usage**
# - Lower `p_local` toward 0, favoring broad jumps
#   - Observe the acceptance rate in the trace title drop: broad proposals
#     land in low-probability regions more often and get rejected
# - Set `p_local` to 1.0
#   - Observe the chain behaves like Cell 3.3's Gibbs sampler: Gibbs is
#     the special case of Metropolis-Hastings where every proposal is a
#     local move drawn from the exact conditional, so it is always accepted

# %% [markdown]
# **Implementation** `cell3_4_metropolis_hastings_widget()`
# - Proposes a local or broad move with probability `p_local`, and accepts
#   or rejects it via the Hastings ratio in `_mh_chain()`
# - Tracks the running posterior estimate for the fixed query
#   $P(Rain \mid Sprinkler{=}T)$ after discarding `burnin` iterations

# %%
hintros.print_obj_info(utils.cell3_4_metropolis_hastings_widget)
