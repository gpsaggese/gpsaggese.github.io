---
jupytext:
  formats: ipynb,py:percent,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

## Imports

+++ {"jp-MarkdownHeadingCollapsed": true}

### Install packages

```{code-cell} ipython3
!sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet jupyterlab-vim)"
!jupyter labextension enable
```

```{code-cell} ipython3
!sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet graphviz)"
```

```{code-cell} ipython3
!sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet dataframe_image)"
```

```{code-cell} ipython3
!sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet jupyterlab-hide-code)"
```

### Import modules

```{code-cell} ipython3
%load_ext autoreload
%autoreload 2

import logging

import arviz as az
import pandas as pd
import xarray as xr
import pymc as pm
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import preliz as pz

import ipywidgets as widgets
from IPython.display import display
```

```{code-cell} ipython3
import msml610_utils as ut

ut.config_notebook()
```

+++ {"heading_collapsed": true}

# Probability distributions

+++

## Bernoulli

+++

- A **Bernoulli variable** is a random variable that takes only two possible values.
  - Typically, these values are $1$ (success) and $0$ (failure).

- **Definition:**
  - $X \sim \text{Bernoulli}(p)$ means $P(X = 1) = p$ and $P(X = 0) = 1 - p$
  - The parameter $p$ represents the probability of success, where $0 \leq p \leq 1$.

- **Intuition:**
  - Represents a single trial of an experiment that can result in one of two outcomes.
  - Examples:
    - Coin flip: $X = 1$ if heads, $X = 0$ if tails.
    - Answer correctness: $X = 1$ if correct, $X = 0$ if incorrect.

```{code-cell} ipython3
np.random.seed(42)

n = 4
p = 0.35

data = stats.bernoulli.rvs(p=p, size=n)
print(data)
```

```{code-cell} ipython3
# Set random seed for reproducibility
np.random.seed(42)

# Define interactive function
def sample_bernoulli(n=4, p=0.35):
    data = stats.bernoulli.rvs(p=p, size=n)
    print(f"Bernoulli(p={p}) - {n} realizations:")
    print(data)

# Create interactive sliders
widgets.interact(
    sample_bernoulli,
    n=widgets.IntSlider(value=4, min=1, max=50, step=1, description='n (samples)'),
    p=widgets.FloatSlider(value=0.35, min=0.0, max=1.0, step=0.01, description='p (success prob)')
);
```

## Binomial

+++

A **binomial random variable** represents the number of successes in a fixed number of independent trials, where each trial has two possible outcomes: success or failure.

- **Parameters:**
  - $n$: number of trials  
  - $p$: probability of success in each trial

- **Probability formula:**
  $$
  P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}
  $$
  where $k = 0, 1, 2, \dots, n$

- **Example:**
  - If you flip a fair coin 10 times, the number of heads follows a `Binomial(10, 0.5)` distribution

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
#?stats.binom
```

```{code-cell} ipython3
np.random.seed(42)

# Create a Binomial.
n = 8
p = 0.01
X = stats.binom(n, p)

# Print k realizations.
k = 4
x = X.rvs(k)
print(x)
```

```{code-cell} ipython3
# Set random seed for reproducibility.
np.random.seed(42)

# Define interactive function.
def sample_binomial(n: int, p: float, k: int) -> None:
    X = stats.binom(n, p)
    x = X.rvs(k)
    print(f"Binomial(n={n}, p={p}) - {k} realizations:")
    print(x)
    

# Create interactive controls.
widgets.interact(
    sample_binomial,
    n=widgets.IntSlider(value=8, min=1, max=100, step=1, description='n (trials)'),
    p=widgets.FloatSlider(value=0.01, min=0.0, max=1.0, step=0.01, description='p (success prob)'),
    k=widgets.IntSlider(value=4, min=1, max=50, step=1, description='k (samples)')
);
```

```{code-cell} ipython3
ut.plot_binomial()
```

```{code-cell} ipython3
params = {
    #"kind": "cdf",
    "kind": "pdf",
    "pointinterval": False,
    "interval": "hdi",   # Highest density interval.
    #"interval": "eti",  # Equal tailed interval.
    "xy_lim": "auto"
}

#help(pz.Binomial.plot_interactive)

# Probability of k successes on N trial flipping a coin with p success
pz.Binomial(p=0.5, n=5).plot_interactive(**params)
```

## Beta

- Continuous prob distribution defined in [0, 1]
- It is useful to model probability or proportion
    - E.g., the probability of success in a Bernoulli trial

- alpha represents "success" parameter
- beta represents "failure" parameter
    - When alpha is larger than beta the distribution skews toward 1, indicating a higher probability of success
    - When alpha = beta the distribution is symmetric and centered around 0.5

```{code-cell} ipython3
np.random.seed(123)

trials = 4
theta = 0.35

# Generate some values.
data = stats.bernoulli.rvs(p=theta_real, size=trials)
print(data)
```

```{code-cell} ipython3
ut.plot_beta()
```

```{code-cell} ipython3
params = {
    #"kind": "cdf",
    "kind": "pdf",
    "pointinterval": False,
    "interval": "hdi",   # Highest density interval.
    #"interval": "eti",  # Equal tailed interval.
    "xy_lim": "auto"
}

alpha = 3.0
beta = 1.0

pz.Beta(alpha=alpha, beta=beta).plot_interactive(**params)
```

# Coin problem: analytical solution

```{code-cell} ipython3
ut.update_prior()
```

## Coin problem: PyMC solution

```{code-cell} ipython3
np.random.seed(123)
n = 4
# Unknown value.
theta_real = 0.35

# Generate some observational data.
data1 = stats.bernoulli.rvs(p=theta_real, size=n)
data1
```

```{code-cell} ipython3
with pm.Model() as model1:
    # Prior.
    theta = pm.Beta('theta', alpha=1., beta=1.)
    # Likelihood.
    y = pm.Bernoulli('y', p=theta, observed=data1)
    # (Numerical) Inference to estimate the posterior distribution through samples.
    idata1 = pm.sample(1000, random_seed=123)
```

```{code-cell} ipython3
az.plot_trace(idata1);
```

```{code-cell} ipython3
#?az.summary
```

```{code-cell} ipython3
az.summary(idata1, kind="stats")
```

```{code-cell} ipython3
az.plot_trace(idata1, kind="rank_bars", combined=True);
```

```{code-cell} ipython3
az.plot_posterior(idata1);
```

## More data

```{code-cell} ipython3
np.random.seed(123)
n = 20
# Unknown value.
theta_real = 0.35

# Generate some observational data.
data2 = stats.bernoulli.rvs(p=theta_real, size=n)
data2
```

```{code-cell} ipython3
with pm.Model() as model2:
    # Prior.
    theta = pm.Beta('theta', alpha=1., beta=1.)
    # Likelihood.
    y = pm.Bernoulli('y', p=theta, observed=data2)
    # (Numerical) Inference to estimate the posterior distribution through samples.
    idata2 = pm.sample(1000, random_seed=123)
```

```{code-cell} ipython3
az.summary(idata2, kind="stats")
```

```{code-cell} ipython3
az.plot_posterior(idata2);
```

## Even more data

```{code-cell} ipython3
np.random.seed(123)
n = 100
# Unknown value.
theta_real = 0.35

# Generate some observational data.
data3 = stats.bernoulli.rvs(p=theta_real, size=n)
data3
```

```{code-cell} ipython3
with pm.Model() as model3:
    # Prior.
    theta = pm.Beta('theta', alpha=1., beta=1.)
    # Likelihood.
    y = pm.Bernoulli('y', p=theta, observed=data3)
    # (Numerical) Inference to estimate the posterior distribution through samples.
    idata3 = pm.sample(1000, random_seed=123)
```

```{code-cell} ipython3
az.summary(idata3, kind="stats")
```

```{code-cell} ipython3
az.plot_posterior(idata3);
```

## Savage-Dickey ratio

```{code-cell} ipython3
for idata in [idata1, idata2, idata3]:
    az.plot_bf(idata, var_name="theta", prior=np.random.uniform(0, 1, 10000), ref_val=0.5);
    plt.xlim(0, 1);
```

## ROPE

```{code-cell} ipython3
for idata in [idata1, idata2, idata3]:
    az.plot_posterior(idata, rope=[0.45, .55], ref_val=0.5)
    plt.xlim(0, 1);
```
