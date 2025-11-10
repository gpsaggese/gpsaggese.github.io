import arviz as az
import pymc as pm
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
#import preliz as pz

np.random.seed(123)
n = 4
# Unknown value.
theta_real = 0.35
data = stats.bernoulli.rvs(p=theta_real, size=n)

with pm.Model() as our_first_model:
    # a priori
    theta = pm.Beta('theta', alpha=1., beta=1.)
    # likelihood
    y = pm.Bernoulli('y', p=theta, observed=data)
    trace = pm.sample(1000, random_seed=123)
