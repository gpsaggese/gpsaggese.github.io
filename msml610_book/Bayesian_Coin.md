---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++

# Bayesian Coin


```{code-cell} ipython3
:tags: [remove-cell]
!sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet jupyterlab-vim)"
!jupyter labextension enable
```

```{code-cell} ipython3
:tags: [remove-cell]
!sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet graphviz)"
```

```{code-cell} ipython3
:tags: [remove-cell]
!sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet dataframe_image)"
```

### Import modules

```{code-cell} ipython3
:tags: [remove-cell]
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
```

```{code-cell} ipython3
:tags: [remove-cell]
import msml610_utils as ut

ut.config_notebook()
```

+++ {"heading_collapsed": true}

# Chap1: Thinking probabilistically

+++

## Binomial

Probability of $k$ heads out of $n$ tosses given bias $p$

\begin{align*}
  & X \sim Binomial(n, p) \\
  & \Pr(k) = \frac{n!}{k! (n - k)!} p^k (1 - p)^{n-k} \\
\end{align*}

```{code-cell} ipython3
#help(pz.Binomial.plot_interactive)
```

```{code-cell} ipython3
np.random.seed(42)

# Create a Normal Gaussian.
n = 8
#p = 0.25
p = 0.01
X = stats.binom(n, p)

# Print 3 realizations.
x = X.rvs(n)
print(x)
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
# Unknown value.
theta_real = 0.35

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

```{code-cell} ipython3
from IPython.display import Code
import inspect
func = ut.update_prior
code = inspect.getsource(func)
#display(Code(code))
```

```{code-cell} ipython3
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from IPython.core.display import HTML

formatter = HtmlFormatter(style="default", full=True, cssclass="codehilite")
highlighted_code = highlight(code, PythonLexer(), formatter)
display(HTML(highlighted_code))
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
