# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lesson 05.1: Machine Learning Theories
#
# **Course**: MSML610: Advanced Machine Learning
#
# **Instructor**: Dr. GP Saggese
#
# **References**:
# - Abu-Mostafa et al.: _"Learning From Data"_ (2012)

# %% [markdown]
# ## Imports

# %%

import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, FloatSlider, IntSlider, fixed

# Set plotting style.
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %% [markdown]
# ## 1. Is Machine Learning Even Possible?
#
# In this section, we explore the fundamental question: can we learn anything from a limited training set?

# %% [markdown]
# ### 1.1 A Simple Visual ML Experiment
#
# We begin with a simple supervised classification problem using 9-bit vectors represented as 3x3 arrays.
#
# **Key Questions**:
# - Given a training set with examples labeled as $f(\mathbf{x}) = -1$ or $f(\mathbf{x}) = +1$, can we predict the label for a new test pattern?
# - Multiple models can fit the same training data but give different predictions on test data
# - Which model is correct?

# %% [markdown]
# ### 1.2 Possible vs Probable
#
# **The Challenge**:
# - A function can assume **any value outside the training data**
# - Example: summer temperature data tells us nothing guaranteed about winter temperatures
#
# **Key Distinction**:
# - **Possible**: Without additional knowledge, the unknown function could behave in any way outside the known data (linear, quadratic, sine wave, etc.)
# - **Probable**: With domain knowledge or historical patterns, we can make reasonable predictions about unknown points
#
# Machine learning relies on moving from "possible" to "probable" statements.

# %% [markdown]
# ### 1.3 Supervised Learning: Bin Analogy (Part 1)
#
# **The Setup**:
# - Consider a bin with red and green marbles
# - We want to estimate $\mu = \Pr(\text{pick a red marble})$ where $\mu$ is unknown
# - We pick $N$ marbles independently with replacement
# - The fraction of red marbles in our sample is $\nu$
#
# **Question**: Does the sample frequency $\nu$ tell us anything about the true frequency $\mu$?
#
# **Answer**:
# - **"No"** (strictly): We don't know anything certain about the marbles we didn't pick. The sample could be mostly green while the bin is mostly red. This is _possible_.
# - **"Yes"** (practically): Under certain conditions, the sample frequency is likely close to the real frequency. This is _probable_.

# %% [markdown]
# ### 1.4 Hoeffding Inequality
#
# The Hoeffding inequality formalizes the intuition that sample statistics are probably close to population statistics.
#
# **Statement**:
# For a Bernoulli random variable $X$ with probability of success $\mu$, if we estimate the mean using $N$ samples with $\nu = \frac{1}{N} \sum_i X_i$, then:
#
# $$
# \Pr(|\nu - \mu| > \varepsilon) \le \frac{2}{e^{2 \varepsilon^2 N}}
# $$
#
# **Key Properties**:
# - Valid for all $N$ and $\varepsilon$ (not asymptotic)
# - Requires random sampling in the same way for both $\nu$ and $\mu$
# - Exponentially small probability that $\nu$ deviates from $\mu$ by more than $\varepsilon$ as $N$ increases
# - Does not depend on $\mu$
# - Trade-off: smaller $\varepsilon$ requires larger $N$ for the same probability bound
#
# This is a **Probably Approximately Correct (PAC)** statement.

# %% [markdown]
# ### Interactive Visualization: Hoeffding Inequality

# %%
# Interactive visualization showing how sample statistics track population statistics.
import utils_Lesson05_Learning_Theory as utils

interact(
    utils.plot_hoeffding_interactive,
    mu=FloatSlider(
        min=0.1, max=0.9, step=0.05, value=0.6, description="mu (true prop)"
    ),
    N=IntSlider(
        min=10, max=1000, step=10, value=100, description="N (sample size)"
    ),
    epsilon=FloatSlider(
        min=0.01,
        max=0.3,
        step=0.01,
        value=0.1,
        description="epsilon (tolerance)",
    ),
    n_trials=IntSlider(
        min=100, max=10000, step=100, value=1000, description="n_trials"
    ),
    figsize=fixed((20, 5)),
)
# Adjust the sliders to see how:
# - Larger N makes the bound tighter (exponential improvement)
# - Smaller epsilon requires larger N for same confidence
# - Empirical violation rate is typically much less than the bound
# - The bound works regardless of the true value of mu

# %% [markdown]
# # Hoeffding Inequality: Study

# %% [markdown]
# ### Cell 1: Empirical vs Expected Distribution

# %%
# Create Bernoulli binomial with probability mu, sample N times, and compare empirical vs expected distribution.
interact(
    utils.plot_hoeffding_study_empirical_vs_expected,
    mu=FloatSlider(
        min=0.1, max=0.9, step=0.05, value=0.6, description="mu (true prob)"
    ),
    N=IntSlider(
        min=10, max=500, step=10, value=100, description="N (sample size)"
    ),
    n_trials=IntSlider(
        min=100, max=5000, step=100, value=1000, description="n_trials"
    ),
    figsize=fixed((20, 5)),
)
# The empirical distribution of nu converges to the expected normal distribution as n_trials increases.

# %% [markdown]
# ### Cell 2: Distribution of mu - nu

# %%
# Compute and visualize the distribution of the difference mu - nu.
interact(
    utils.plot_hoeffding_study_difference_distribution,
    mu=FloatSlider(
        min=0.1, max=0.9, step=0.05, value=0.6, description="mu (true prob)"
    ),
    N=IntSlider(
        min=10, max=500, step=10, value=100, description="N (sample size)"
    ),
    n_trials=IntSlider(
        min=100, max=5000, step=100, value=1000, description="n_trials"
    ),
    figsize=fixed((20, 5)),
)
# The distribution of mu - nu is centered at zero and its spread decreases with larger N.

# %% [markdown]
# ### 1.5 Supervised Learning: Bin Analogy (Part 2)
#
# **Connecting to Machine Learning**:
#
# | Bin Analogy | Machine Learning |
# |-------------|------------------|
# | Each marble is a point $\mathbf{x} \in \mathcal{X}$ | Point in input space |
# | Red marble = correct prediction | $h(\mathbf{x}) = f(\mathbf{x})$ |
# | Green marble = incorrect prediction | $h(\mathbf{x}) \neq f(\mathbf{x})$ |
# | Sample frequency $\nu$ | In-sample error $E_{in}(h)$ |
# | Population frequency $\mu$ | Out-of-sample error $E_{out}(h)$ |
#
# **Result**: Hoeffding inequality bounds the generalization error:
#
# $$
# \Pr(|E_{in} - E_{out}| > \varepsilon) \le c
# $$
#
# **Conclusion**: Generalization to unknown points is possible. **Machine learning is possible!**

# %% [markdown]
# ### 1.6 Validation vs Learning
#
# **Validation Setup**:
# - Given a **fixed** hypothesis $h$
# - Hoeffding tells us that $E_{in}(h)$ is probably close to $E_{out}(h)$
# - This validates that our model generalizes
#
# **Learning Setup**:
# - Choose the **best** hypothesis from $M$ hypotheses: $h \in \mathcal{H} = \{h_1, \ldots, h_M\}$
# - Need a bound that works for the chosen hypothesis, regardless of which one we pick
# - Using the union bound:
#
# \begin{align*}
# \Pr(|E_{in}(g) - E_{out}(g)| > \varepsilon) &\le \Pr\left(\bigcup_{i=1}^M |E_{in}(h_i) - E_{out}(h_i)| > \varepsilon\right) \\
# &\le \sum_{i=1}^M \Pr(|E_{in}(h_i) - E_{out}(h_i)| > \varepsilon) \\
# &\le 2M \exp(-2\varepsilon^2 N)
# \end{align*}
#
# **Problem**: The bound is weak because $M$ can be very large (or infinite).

# %% [markdown]
# ### 1.7 Validation vs Learning: Coin Analogy
#
# **Validation (Single Coin)**:
# - Have one coin, want to determine if it's fair
# - Assume $\mu = 0.5$ (unbiased)
# - Toss 10 times
# - Probability of getting 10 heads (appears biased with $\nu = 1.0$):
#   $$\Pr(\nu = 1.0) = 1/2^{10} \approx 0.1\%$$
# - **Conclusion**: Very unlikely that out-of-sample behavior differs significantly from in-sample
#
# **Learning (Many Coins)**:
# - Have 1000 fair coins, need to choose one
# - Probability that at least one appears totally biased (10 heads in 10 tosses):
#   $$\Pr(\text{at least one } \nu = 1.0) = 1 - (1 - 1/2^{10})^{1000} \approx 63\%$$
# - **Conclusion**: More than 50% chance that we find a coin that looks biased!
#
# This illustrates why the learning bound is weaker than the validation bound.

# %% [markdown]
# ### 1.8 Why the Union Bound Is Weak
#
# The union bound:
# $$\Pr(|E_{in} - E_{out}| > \varepsilon) \le 2M \exp(-2\varepsilon^2 N)$$
#
# is **artificially too loose** because:
#
# - The union bound assumes "bad events" $\mathcal{B}_i$ (where hypothesis $h_i$ doesn't generalize) are disjoint
# - **In reality**, bad events are extremely overlapping because similar hypotheses fail in similar ways
# - Similar hypotheses (e.g., two perceptrons with similar weights) make similar mistakes on similar data points
# - The union bound counts overlapping events multiple times, leading to a conservative estimate
#
# This motivates the need for a tighter bound based on the **effective number of hypotheses**.

# %% [markdown]
# ### 1.9 Training vs Testing: College Course Analogy
#
# Machine learning phases parallel studying for a college course:
#
# | ML Phase | College Course Equivalent |
# |----------|---------------------------|
# | **Learning Phase** (Training Set) | Studying the course material |
# | **Validation Phase** (Validation Set) | Practice problems with solutions - helps identify weaknesses |
# | **Testing Phase** (Test Set) | Final exam - different from practice, gauges true learning |
# | **Out-of-Sample Phase** (Production) | Using knowledge on the job after graduation |
#
# **Key Insights**:
# - The goal isn't to do well on the exam (test set), but to actually learn (generalize)
# - Giving out exam problems in advance wouldn't gauge learning effectively (data snooping)
# - What ultimately matters is real-world performance (out-of-sample)

# %% [markdown]
# ## 2. Growth Function
#
# To get a tighter bound than the union bound, we need to count the **effective number** of hypotheses rather than the total number $M$.

# %% [markdown]
# ### 2.1 Dichotomy: Definition
#
# **Setup**:
# - Classify $N$ fixed points $\mathbf{x}_1, \ldots, \mathbf{x}_N$ using hypothesis set $\mathcal{H}$
# - Consider an assignment $D$ of points to classes: $\mathbf{d}_1, \ldots, \mathbf{d}_N$
#
# **Definition**: $D$ is a **dichotomy** for $\mathcal{H}$ if and only if there exists $h \in \mathcal{H}$ that achieves the classification $D$.
#
# **Example: 4 points in a plane with 2D perceptrons**:
# - Different positions of the separating hyperplane create different dichotomies
# - There are at most $2^N$ possible dichotomies
# - Certain classifications are impossible (e.g., XOR pattern for linearly separable data)
# - For 4 points: perceptrons can achieve 14 out of 16 possible dichotomies

# %% [markdown]
# ### 2.2 Dichotomies vs Hypotheses
#
# **Hypothesis**: Classifies every point in $\mathcal{X}$: $\mathcal{X} \rightarrow \{-1, +1\}$
#
# **Dichotomy**: Classifies only a fixed set of points: $\{\mathbf{x}_1, \ldots, \mathbf{x}_N\} \rightarrow \{-1, +1\}$
# - Dichotomies are "mini-hypotheses" (hypotheses restricted to given points)
#
# **Key Differences**:
# - Number of hypotheses: Usually infinite ($|\mathcal{H}| = \infty$)
# - Number of dichotomies: Always finite ($|\mathcal{H}(\mathbf{x}_1, \ldots, \mathbf{x}_N)| \le 2^N$)
#
# **What Determines a Dichotomy**:
# - Number of points $N$
# - Hypothesis set $\mathcal{H}$ (possible models)
# - Where points are placed
# - How points are assigned to classes
#
# **From the training set perspective**:
# - What matters are dichotomies, not hypotheses
# - Many (infinite) hypotheses can correspond to the same dichotomy
# - The "complexity" of $\mathcal{H}$ relates to the number of achievable dichotomies

# %% [markdown]
# ### 2.3 Growth Function: Definition
#
# The **growth function** counts the maximum number of dichotomies on $N$ points:
#
# $$
# m_{\mathcal{H}}(N) = \max_{\mathbf{x}_1, \ldots, \mathbf{x}_N \in \mathcal{X}} |\mathcal{H}(\mathbf{x}_1, \ldots, \mathbf{x}_N)|
# $$
#
# **Why use the maximum?**
# - Dichotomies depend on point distribution and assignment
# - Growth function considers the "most favorable" placement for the hypothesis set
# - Provides a worst-case bound on the effective number of hypotheses
#
# **Computing $m_{\mathcal{H}}(N)$ by brute force**:
# 1. Consider all possible placements of $N$ points
# 2. Consider all possible class assignments for these points
# 3. For each hypothesis $h \in \mathcal{H}$, compute the resulting dichotomy
# 4. Count the number of unique dichotomies

# %% [markdown]
# ### 2.4 Growth Function: Properties
#
# **Growth function increases with $N$**:
# - $m_{\mathcal{H}}(N)$ increases (not always monotonically) with $N$
# - Can ignore additional points to get same classification, so $m_{\mathcal{H}}(N) \ge m_{\mathcal{H}}(N-1)$
#
# **Growth function increases with complexity**:
# - More complex $\mathcal{H}$ (more flexible models) → larger $m_{\mathcal{H}}(N)$
# - Higher dimensional input space → larger $m_{\mathcal{H}}(N)$

# %% [markdown]
# ### 2.5 Growth Function: Examples
#
# **1. Perceptron on a Plane**:
# - $m_{\mathcal{H}}(3) = 8 = 2^3$ (can shatter 3 points)
# - $m_{\mathcal{H}}(4) = 14 < 2^4$ (cannot achieve XOR patterns)
#
# **2. Positive Rays** on $\mathbb{R}$: $h(x) = \text{sign}(x - a)$
# - $m_{\mathcal{H}}(N) = N + 1$
# - The threshold $a$ can be placed in $N+1$ intervals created by $N$ points
#
# **3. Positive Intervals** on $\mathbb{R}$: $h(x) = 1$ if $x \in [a,b]$, else $-1$
# - $m_{\mathcal{H}}(N) = \binom{N+1}{2} + 1 \sim N^2$
# - Choose 2 endpoints from $N+1$ intervals, plus the "all negative" case
#
# **4. Convex Sets on a Plane**:
# - $m_{\mathcal{H}}(N) = 2^N$
# - Place points on a circle; any subset can be enclosed by a convex polygon
# - Can shatter any number of points (infinite VC dimension)

# %% [markdown]
# ### 2.6 Break Point of a Hypothesis Set
#
# **Shattering**: A hypothesis set $\mathcal{H}$ **shatters $N$ points** if and only if $m_{\mathcal{H}}(N) = 2^N$
# - There exists some arrangement of $N$ points where all $2^N$ classifications are achievable
# - Doesn't mean all arrangements of $N$ points can be shattered
#
# **Break Point**: $k$ is a **break point** for $\mathcal{H}$ if and only if $m_{\mathcal{H}}(k) < 2^k$
# - No data set of size $k$ can be shattered by $\mathcal{H}$
#
# **Examples**:
# - **2D Perceptron**: break point is 4 (cannot shatter 4 points)
# - **Positive rays**: break point is 2 (cannot shatter 2 points)
# - **Positive intervals**: break point is 3 (cannot shatter 3 points)
# - **Convex sets**: no break point (can shatter any number of points)

# %% [markdown]
# ### 2.7 Break Point and Learning
#
# **Key Result**: If there exists a break point for $\mathcal{H}$, then:
#
# 1. **Growth function is polynomial**: $m_{\mathcal{H}}(N)$ is polynomial in $N$
#
# 2. **Vapnik-Chervonenkis (VC) Inequality**: Instead of Hoeffding's bound
#    $$\Pr(|E_{in}(g) - E_{out}(g)| > \varepsilon) \le 2M e^{-2\varepsilon^2 N}$$
#
#    we get:
#    $$\Pr(\text{bad generalization}) \le 4 m_{\mathcal{H}}(2N) e^{-\frac{1}{8}\varepsilon^2 N}$$
#
# 3. **Generalization**: Since $m_{\mathcal{H}}(N)$ is polynomial, it's dominated by the negative exponential for large enough $N$
#
# **Conclusion**: A hypothesis set can be characterized by the **existence and value of a break point**. With a break point, machine learning works!

# %% [markdown]
# ## 3. The VC Dimension
#
# The VC (Vapnik-Chervonenkis) dimension provides a single number that characterizes the complexity of a hypothesis set.

# %% [markdown]
# ### 3.1 VC Dimension: Definition
#
# The **VC dimension** of a hypothesis set $\mathcal{H}$, denoted $d_{VC}(\mathcal{H})$, is the **largest value of $N$** for which $m_{\mathcal{H}}(N) = 2^N$.
#
# - I.e., the VC dimension is the maximum number of points $\mathcal{H}$ can shatter
#
# **Properties**: If $d_{VC}(\mathcal{H}) = d$, then:
#
# 1. **Existence**: There exists some arrangement of $d$ points that can be shattered by $\mathcal{H}$
#    - Not all sets of $d$ points can be shattered
#    - Random placement of $d$ points may not be shatterable
#
# 2. **No larger shattering**: Cannot shatter $d+1$ points in any arrangement
#
# 3. **Smaller sets**: $\mathcal{H}$ can shatter $N$ points for any $N \le d_{VC}$
#
# 4. **Break point**: The smallest break point is $d_{VC} + 1$
#
# 5. **Growth function bound**: $m_{\mathcal{H}}(N) \le \sum_{i=0}^{d_{VC}} \binom{N}{i}$
#
# 6. **Polynomial order**: $d_{VC}$ is the order of the polynomial bounding $m_{\mathcal{H}}$

# %% [markdown]
# ### 3.2 VC Dimension: Interpretation
#
# **VC dimension measures complexity** in terms of **effective parameters**.
#
# **Key Insights**:
#
# 1. **Often equals number of parameters**:
#    - A perceptron in $d$-dimensional space has $d_{VC} = d + 1$
#    - This equals the number of parameters (weights)!
#
# 2. **Black box measure**:
#    - Estimates effective parameters by counting shatterable points
#    - Doesn't require inspecting the model's internals
#
# 3. **Not all parameters are effective**:
#    - Combining $N$ 1D perceptrons gives $2N$ parameters
#    - But effective degrees of freedom remain 2
#    - Some parameters may be redundant or constrained
#
# 4. **Implications for training**:
#    - More complex $\mathcal{H}$ (higher $d_{VC}$) → more parameters
#    - More parameters → requires more training examples

# %% [markdown]
# ### 3.3 VC Generalization Bounds
#
# **Question**: How many data points $N$ are needed to ensure $\Pr(|E_{in} - E_{out}| > \varepsilon) \le \delta$?
#
# **VC Inequality**:
# $$\Pr(\text{bad generalization}) \le 4 m_{\mathcal{H}}(2N) e^{-\frac{1}{8}\varepsilon^2 N}$$
#
# **Behavior**: The bound behaves like $N^d e^{-N}$:
# - For small $N$: polynomial term $N^d$ dominates (bound is loose)
# - For large $N$: exponential term $e^{-N}$ dominates (bound approaches 0)
# - Larger $d$ (more complex models) requires larger $N$ to reach the useful region
#
# **Rule of Thumb**:
# $$N \ge 10 \cdot d_{VC}$$
# for reasonable generalization guarantees.

# %% [markdown]
# ### 3.4 Using the VC Bound
#
# The VC inequality can be rearranged to answer different questions:
#
# **Given $\varepsilon$ and $\delta$, find required $N$**:
# - "To get 1% error with 95% confidence, how many examples do I need?"
#
# **Given $N$ and $\delta$, find achievable $\varepsilon$**:
# - "With 1000 examples, what error can I achieve with 95% confidence?"
#
# **Generalization bound**: Setting $\delta = 4 m_{\mathcal{H}}(2N) e^{-\frac{1}{8}\varepsilon^2 N}$ and solving for $\varepsilon$:
#
# $$\Omega(N, \mathcal{H}, \delta) = \sqrt{\frac{8}{N} \ln \frac{4 m_{\mathcal{H}}(2N)}{\delta}}$$
#
# Then with probability $\ge 1 - \delta$:
# $$E_{out} \le E_{in} + \Omega(N, \mathcal{H}, \delta)$$
#
# **Interpretation**: Out-of-sample error is bounded by in-sample error plus a complexity penalty that decreases with more data.

# %% [markdown]
# ### 3.5 How to Void the VC Analysis Guarantee
#
# **Scenario**: Data is genuinely non-linear (e.g., circles in center, crosses in corners)
#
# **Approach**: Transform to higher-dimensional space $\mathcal{Z}$ where data becomes linearly separable:
# $$\Phi: \mathbf{x} = (x_0, \ldots, x_d) \rightarrow \mathbf{z} = (z_0, \ldots, z_{\tilde{d}})$$
#
# **The Trap**: Progressively refining the transformation:
# - Start with: $\mathbf{z} = (1, x_1, x_2, x_1 x_2, x_1^2, x_2^2)$
# - "Simplify" to: $\mathbf{z} = (1, x_1^2, x_2^2)$
# - "Simplify" to: $\mathbf{z} = (1, x_1^2 + x_2^2)$
# - "Optimize" to: $\mathbf{z} = (x_1^2 + x_2^2 - 0.6)$
#
# **What went wrong?**
# - Each "simplification" was based on examining the data
# - Setting coefficients to zero or choosing specific transformations based on data is **data snooping**
# - The effective $d_{VC}$ is that of the **initial hypothesis set** before simplification, not the final model
#
# **Key Principle**: VC analysis is a warranty, **forfeited if data is examined before model selection**.
# - Once you peek at the data to guide model selection, you've effectively searched through a much larger hypothesis space
# - The complexity penalty should reflect the full search space, not just the final model

# %% [markdown]
# ## Summary
#
# **Is machine learning possible?**
# - Yes! Through the lens of "probable" rather than "possible"
# - Hoeffding inequality shows in-sample performance probably tracks out-of-sample performance
#
# **Growth function and dichotomies**:
# - Effective number of hypotheses (dichotomies) is much smaller than total number
# - Growth function $m_{\mathcal{H}}(N)$ counts maximum achievable dichotomies
# - Break point indicates when $m_{\mathcal{H}}(N)$ becomes polynomial
#
# **VC dimension**:
# - Single number characterizing hypothesis set complexity
# - Maximum number of points the hypothesis set can shatter
# - Often relates to number of parameters
# - Determines sample complexity: need $N \ge 10 \cdot d_{VC}$ examples
#
# **VC inequality**:
# - Provides generalization bounds: $E_{out} \le E_{in} + \Omega(N, \mathcal{H}, \delta)$
# - Complexity penalty $\Omega$ decreases with more data
# - Warranty is void if you peek at data during model selection (data snooping)
