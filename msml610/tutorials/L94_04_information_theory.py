# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style.
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %%
import msml610_utils as ut
import L94_04_information_theory_utils as utils

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)


# %% [markdown]
# # Cell 1: Entropy and Uncertainty
#
# **Entropy** $H(X)$ of a discrete random variable $X$ is defined as:
#
# $$H(X) = -\sum_x p(x) \log_2 p(x)$$
#
# - Entropy quantifies the average level of **information**, **surprise**, or **uncertainty** inherent in the variable's possible outcomes
# - High entropy = more unpredictability
# - Low entropy = more certainty

# %%
# Test with fair coin.
# Two equally likely outcomes → maximum uncertainty, $H = 1$ bit.
fair_coin = [0.5, 0.5]
print(f"Fair coin entropy: {utils.cell1_calculate_entropy(fair_coin):.4f} bits")

# %%
# Test with biased coin.
# If heads occurs 90% of the time → less uncertainty, $H < 1$ bit.
biased_coin = [0.9, 0.1]
print(
    f"Biased coin (90-10) entropy: {utils.cell1_calculate_entropy(biased_coin):.4f} bits"
)

# %%
# Test with broken coin.
biased_coin = [1.0, 0.0]
print(
    f"Biased coin (100-0) entropy: {utils.cell1_calculate_entropy(biased_coin):.4f} bits"
)
# If heads occurs 100% of the time → no uncertainty, $H = 0$ bit.

# %% [markdown]
# # Cell 2: Entropy vs Variance
#
# Entropy and variance are related but measure different properties:
# - **Variance** measures how far values are from the mean
# - **Entropy** measures how unpredictable a random draw is
#
# A distribution can have high variance but low entropy, or vice versa.

# %%
# Compare two distributions with the same variance but different entropy.
# Distribution 1: Bimodal with peaks at extremes.
dist1_values = np.array([5 - np.sqrt(2), 5 + np.sqrt(2)])
dist1_probs = np.array([0.5, 0.5])

file_name = "figures/Lesson94_Bimodal_Distribution.png"
utils.cell2_plot_distribution_with_stats(
    values=dist1_values,
    probabilities=dist1_probs,
    title="Distribution 1: Two peaks at extremes",
    save_fig=file_name,
)
# This distribution has high variance (spread) but low entropy (only 1 bit).

# %%
# Distribution 2: Uniform over middle values.
dist2_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
dist2_probs = np.array([0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0])

file_name = "figures/Lesson94_Uniform_Distribution.png"
utils.cell2_plot_distribution_with_stats(
    values=dist2_values,
    probabilities=dist2_probs,
    title="Distribution 2: Uniform over middle values",
    save_fig=file_name,
)
# This distribution has lower variance but higher entropy (~2.32 bits).

# %% [markdown]
# ## Cell 2.1: Entropy and Distribution Spread
#
# - Generally, more spread in a distribution leads to higher entropy, but there are exceptions
# - Increasing the support of a uniform distribution increases variance but not entropy
# - The relationship depends on the shape of the distribution

# %%
# Example: Uniform distributions with different support.
# Uniform over 2 values.
uniform_2 = np.array([0.5, 0.5])
values_2 = np.array([0, 1])

file_name = "figures/Lesson94_Uniform2.png"
utils.cell2_plot_distribution_with_stats(
    values=values_2,
    probabilities=uniform_2,
    title="Uniform distribution over 2 values",
    save_fig=file_name,
)
# A uniform distribution over 2 outcomes has 1 bit of entropy.

# %%
# Uniform over 4 values.
uniform_4 = np.array([0.25, 0.25, 0.25, 0.25])
values_4 = np.array([0, 1, 2, 3])

file_name = "figures/Lesson94_Uniform4.png"
utils.cell2_plot_distribution_with_stats(
    values=values_4,
    probabilities=uniform_4,
    title="Uniform distribution over 4 values",
    save_fig=file_name,
)
# A uniform distribution over 4 outcomes has 2 bits of entropy.

# %%
# Uniform over 8 values.
uniform_8 = np.array([0.125] * 8)
values_8 = np.array([0, 1, 2, 3, 4, 5, 6, 7])

file_name = "figures/Lesson94_Uniform8.png"
utils.cell2_plot_distribution_with_stats(
    values=values_8,
    probabilities=uniform_8,
    title="Uniform distribution over 8 values",
    save_fig=file_name,
)
# A uniform distribution over 8 outcomes has 3 bits of entropy.

# %% [markdown]
# ## Cell 2.2: Entropy and Uncertainty: Shape Matters
#
# Entropy is closely related to the shape of the probability distribution:
# - **Flat (uniform) distribution** → high entropy, high uncertainty
# - **Sharply peaked distribution** → low entropy, low uncertainty
# - **Multi-modal distribution** → can have high entropy despite low variance

# %%
# Example 1: Flat distribution has high entropy.
flat_dist = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
values_flat = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

file_name = "figures/Lesson94_Flat_Distribution.png"
utils.cell2_plot_distribution_with_stats(
    values=values_flat,
    probabilities=flat_dist,
    title="Flat (uniform) distribution",
    save_fig=file_name,
)
# Uniform distribution has maximum entropy for given number of outcomes.

# %%
# Example 2: Sharply peaked distribution has low entropy.
peaked_dist = np.array(
    [0.00, 0.01, 0.01, 0.01, 0.92, 0.01, 0.01, 0.01, 0.01, 0.01]
)
values_peaked = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

file_name = "figures/Lesson94_Sharply_Peaked_Distribution.png"
utils.cell2_plot_distribution_with_stats(
    values=values_peaked,
    probabilities=peaked_dist,
    title="Sharply peaked distribution (92% at position 4)",
    save_fig=file_name,
)
# Concentrated probability → low uncertainty → low entropy.

# %%
# Example 3: Two close peaks can have low variance but high entropy.
values_two_peaks = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
two_peaks = np.array([0.0, 0.0, 0.45, 0.0, 0.1, 0.0, 0.45, 0.0, 0.0, 0.0])

file_name = "figures/Lesson94_Two_Peaks_Distribution.png"
utils.cell2_plot_distribution_with_stats(
    values=values_two_peaks,
    probabilities=two_peaks,
    title="Distribution with two close peaks (at positions 2 and 6)",
    save_fig=file_name,
)
# Two equally likely peaks → high entropy (~1.1 bits) despite moderate variance.

# %% [markdown]
# # Cell 3: Interactive Visualization: Binary Entropy
#
# - Use the slider below to adjust the probability $p$ of a binary random variable
# - Observe how entropy changes as probability varies
#
# **Parameters**:
# - `Probability p`: Probability of success for binary random variable (between 0 and 1)

# %%
utils.cell3_create_binary_entropy_widget()

# %%
utils.cell3_generate_binary_entropy_animation()

# %% [markdown]
# # Cell 4: Joint Entropy
#
# - **Joint entropy** $H(X, Y)$ of two variables $X$ and $Y$ is defined as:
#   $$H(X, Y) = -\sum_{x,y} p(x,y) \log_2 p(x,y)$$
# - Describes the information needed for the joint distribution of $X$ and $Y$
# - For independent variables: $H(X, Y) = H(X) + H(Y)$
#
# **Parameters**:
# - `Dependence`: Level of dependence between $X$ and $Y$ (0 = independent, 1 = fully dependent)

# %%
utils.cell4_create_joint_entropy_widget()


# %%
utils.cell4_generate_joint_entropy_animation()

# %% [markdown]
# # Cell 5: Conditional Entropy
#
# - **Conditional entropy** $H(Y|X)$ measures uncertainty in $Y$ after observing $X$:
#   $$H(Y|X) = -\sum_{x,y} p(x,y) \log_2 p(y|x) = \sum_x p(x) H(Y|X=x)$$
#
# **Properties:**
# - Low $H(Y|X)$ implies $X$ has strong predictive power for $Y$
# - If $Y = X$, then $H(Y|X) = 0$ (no uncertainty)
# - If $X$ and $Y$ are independent, then $H(Y|X) = H(Y)$
# - **Chain Rule for Entropy**:
#   $$H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$
#
# **Parameters**:
# - `Dependence`: Level of dependence between $X$ and $Y$ (0 = independent, 1 = fully dependent)

# %%
utils.cell5_create_conditional_entropy_widget()

# %%
utils.cell5_generate_conditional_entropy_animation()


# %% [markdown]
# # Cell 6: Mutual Information
#
# - **Mutual information** $I(X;Y)$ measures how much knowing one variable reduces uncertainty about the other:
#   $$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)$$
# - Measures the shared information between two variables
# - Quantifies the reduction in uncertainty about one variable given the other
# - Symmetric: $I(X;Y) = I(Y;X)$
#
# **Parameters**:
# - `Dependence`: Level of dependence between variables (for Venn diagram widgets)
# - `Correlation`: Correlation coefficient between variables (for correlation widget)

# %%
utils.cell6_create_mutual_information_venn_widget()


# %%
utils.cell6_generate_mutual_info_venn_binary_animation()

# %%
utils.cell6_generate_mutual_info_venn_weather_animation()

# %%
utils.cell6_create_mutual_info_correlation_widget()

# %%
utils.cell6_generate_mutual_info_correlation_animation()


# %% [markdown]
# # Cell 7: KL Divergence
#
# - **Kullback-Leibler (KL) Divergence** $D_{KL}(P \| Q)$ measures how one distribution differs from another:
#   $$D_{KL}(P \| Q) = \sum_x P(x) \log_2 \frac{P(x)}{Q(x)}$$
# - Measures the information lost when $Q$ is used to approximate $P$
# - Asymmetric: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$
#
# **Parameters**:
# - `P(outcome=1)`: Probability of outcome 1 in true distribution $P$ (0.05 to 0.95)
# - `Q(outcome=1)`: Probability of outcome 1 in approximate distribution $Q$ (0.05 to 0.95)

# %%
utils.cell7_create_kl_divergence_widget()

# %%
utils.cell7_generate_kl_divergence_animation()


# %% [markdown]
# # Cell 8: Cross-Entropy
#
# - **Cross-entropy** $H(P, Q)$ measures the average number of bits needed to encode data from $P$ using code optimized for $Q$:
#   $$H(P, Q) = -\sum_x P(x) \log_2 Q(x)$$
# - **Relationship**: $H(P, Q) = H(P) + D_{KL}(P \| Q)$$
#
# **Applications:**
# - Loss function in classification (logistic regression, neural networks)
# - Model evaluation and comparison
# - Information compression
#
# **Parameters**:
# - `P(outcome=1)`: Probability of outcome 1 in true distribution $P$ (0.05 to 0.95)
# - `Q(outcome=1)`: Probability of outcome 1 in model distribution $Q$ (0.05 to 0.95)

# %% [markdown]
# **Key insights**:
# - When $P = Q$ (on diagonal), cross-entropy equals entropy $H(P)$ (optimal encoding)
# - When $P \neq Q$, cross-entropy = $H(P) + D_{KL}(P \| Q)$ (extra cost from model mismatch)
# - This extra cost is why cross-entropy works as a loss function in machine learning

# %%
utils.cell8_create_cross_entropy_widget()


# %%
utils.cell8_generate_cross_entropy_animation()

# %% [markdown]
# # Cell 9: Data Processing Inequality
#
# - **Statement**: Processing data cannot increase information, it can only lose information
# - Formally, if $X \to Y \to Z$ forms a Markov chain, then:
#   $$I(X;Z) \leq I(X;Y)$$
# - **Intuition**: Information can only be lost through processing, never gained
# - **Example**: If $X$ is a raw image and $Y$ is compressed version, no further processing $Z$ will recover more information about $X$ than $Y$ already contains
#
# **Parameters**:
# - `Noise Level`: Amount of noise in the data processing pipeline (0.0 to 1.0)

# %% [markdown]
# **Key insights**:
# - Noise Level = 0.0: Clean processing, minimal information loss, $I(X;Z)$ close to $I(X;Y)$
# - Noise Level = 1.0: Maximum noise, substantial information loss, $I(X;Z) \ll I(X;Y)$
# - The inequality $I(X;Z) \leq I(X;Y)$ is always satisfied, demonstrating the fundamental principle
#
# %%
utils.cell9_create_data_processing_inequality_widget()


# %%
utils.cell9_generate_data_processing_inequality_animation()

# %% [markdown]
# # Cell 10: Minimum Description Length (MDL)
#
# - **Principle**: Select the model that minimizes total description length:
#   $$MDL(H) = L(H) + L(D | H)$$
# - Where:
#   - $L(H)$ = length of the model/hypothesis
#   - $L(D|H)$ = length of data encoded using the model
#
# - **Intuition**: Balances model complexity with data fit (Occam's Razor principle)
#
# **Parameters**:
# - `Polynomial Degree`: Degree of polynomial model (1 to 8)
#
# **Key insights**:
# - Low degree (1-2): Simple model, poor fit, high data encoding cost (underfitting)
# - Optimal degree (3-4): Balanced model, minimum total MDL
# - High degree (6-8): Complex model, high model cost, overfitting penalty
# %%
utils.cell10_create_mdl_widget()


# %%
utils.cell10_generate_mdl_animation()

# %% [markdown]
# # Cell 11: Kolmogorov Complexity
#
# - **Definition**: The length of the shortest program that outputs a string $x$
# - **Examples**:
#   - String "00000000..." has low complexity (simple loop)
#   - Random string has high complexity (no pattern, no compression)
# - **Properties**:
#   - Not computable (theoretical concept)
#   - Related to MDL in practice
#   - Measures algorithmic randomness
#
# **Parameters**:
# - `String Type`: Choose between structured patterns (low K-complexity) and random strings (high K-complexity)
# - `Length`: String length to observe how K-complexity scales

# %% [markdown]
# **Key insights**:
# - Patterned strings: Short program generates long output (low K-complexity)
# - Random strings: Must include all bits in description (high K-complexity, incompressible)
# - K-complexity is uncomputable, but compression gives practical approximation

# %%
utils.cell11_create_kolmogorov_complexity_widget()


# %%
utils.cell11_generate_kolmogorov_complexity_animation()
