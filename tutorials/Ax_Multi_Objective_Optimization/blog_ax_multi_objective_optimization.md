# Fewer Experiments, Better Results: Optimizing Marketing Campaigns with Ax

*I recently explored how Meta's Ax library can optimize marketing campaigns with minimal experimentation. Here's what I learned and why you should consider it for your next optimization problem.*

## Why You Should Care About Adaptive Experimentation

Imagine you're running a digital advertising campaign. You have a budget, a bidding strategy with several tunable parameters, and a simple goal: maximize clicks while minimizing costs. The problem? You can't just try every possible combination of parameters, each experiment costs real money and takes real time.

This is a **black-box optimization** problem. The relationship between your input parameters and the outcome (clicks, conversions, revenue) cannot be expressed as a formula. The only way to know how well a set of parameters performs is to run an actual campaign.

Traditional approaches like grid search or random search waste most of their budget on uninformative trials. What if you could intelligently decide which parameters to test next, learning from each experiment to make the next one better?

This is exactly what [Adaptive Experimentation (Ax)](https://ax.dev/) does. Developed by Meta and built on top of [BoTorch](https://botorch.org/) (a Bayesian Optimization framework on PyTorch), Ax provides a high-level API that makes it straightforward to run Bayesian Optimization experiments, even if you've never heard of Gaussian Processes.

In a [previous post](blog_introduction_to_bayesian_optimization.md), I covered the fundamentals of Bayesian Optimization. Here, I'll show you how Ax brings that theory to life in a real-world marketing scenario.

## The Competition: How Does Ax Stack Up?

Several libraries and platforms exist for Bayesian Optimization and hyperparameter tuning. Here's how Ax compares to the most popular ones:

| Library | Strengths | Limitations |
|---------|-----------|-------------|
| **Ax** | High-level API, multi-objective support, Pareto frontier, built on BoTorch/PyTorch, Experiment oriented | Heavier dependency stack (PyTorch) |
| **Optuna** | Lightweight, great for hyperparameter tuning, pruning support | Limited multi-objective capabilities, designed for ML tuning|
| **Hyperopt** | Simple API, TPE-based optimization | No multi-objective, less flexible surrogate models |

What sets Ax apart:

- **Multi-objective optimization** with Pareto frontier computation out of the box
- **Outcome constraints** to enforce real-world limits (e.g., "budget must stay under \$X")
- **Batch trials** suggest multiple experiments to run in parallel
- **Built on BoTorch**, giving you access to state-of-the-art Bayesian Optimization models under the hood
- **Simple, stateful API** that abstracts away the mathematical complexity
- **Real-life application** designed to be used in real-world experiments

## The Problem: Optimizing a Real-Time Bidding Strategy

### How Real-Time Bidding Works

Real-Time Bidding (RTB) is how most digital advertising works today. When a user visits a website, an auction happens in milliseconds:

1. The **Publisher** (website) sends an auction request to the **Ad Exchange**
2. Multiple **DSPs** (Demand Side Platforms), each representing an advertiser, submit bids
3. The highest bid wins, and the ad is displayed
4. The advertiser typically pays the second-highest bid price (second-price auction)

![RTB Process](images/rtb-process-2.png)

### The Bidding Strategy

A DSP needs a **bidding strategy** to decide how much to bid for each impression. In this case, we'll use the **Linear-form bidding of pCTR (Lin)** strategy, one of the most effective and widely used approaches:

$$
\text{final\_bid} = \text{base\_bid} \times \left(1 + c_{\text{ctr}} \times \left(\frac{\text{pCTR}}{\text{avg\_CTR}} - 1\right)\right) \times \left(1 + c_{\text{pay}} \times (\text{pay\_to\_bid\_ratio} - 1)\right) \times \left(1 + c_{\text{pace}} \times (\text{pacing} - 1)\right)
$$

Where:
- **base_bid**: How much the advertiser is willing to pay per thousand impressions of average quality
- **ctr_coefficient**: Coefficient to adjust the bid based on the predicted CTR
- **pCTR**: Predicted CTR for the impression. Defined by a machine learning model.
- **avg_CTR**: Average CTR for the campaign. Defined by the advertiser.
- **pay_to_bid_ratio**: Adjustment factor since the winning price is usually lower than the bid
- **pacing**: Budget pacing coefficient to distribute spending across the day

The **hyperparameters** we need to optimize are:
- `base_bid` (20 to 500)
- `ctr_coefficient` (0 to 1)
- `bid_to_pay_ratio` (0 to 1)
- `pacing_coefficient` (0 to 1)

Manually tuning these parameters by running campaigns is expensive and slow. This is where Ax comes in.

## The Solution: Ax in Action

### Step 1: A Warm-Up with the Hartmann Function

Before tackling the marketing problem, let's see how Ax works on a well-known benchmark, the [Hartmann function](https://en.wikipedia.org/wiki/Hartmann_function) in 6 dimensions:

$$
f(\mathbf{x}) = -\sum_{i=1}^{4} \alpha_i \exp \left( -\sum_{j=1}^{6} A_{ij} (x_j - P_{ij})^2 \right)
$$

> **Note:** We previously explored the basics of Bayesian Optimization and the Hartmann function example in [Introduction to Bayesian Optimization](./blog_introduction_to_bayesian_optimization.md). If you need a gentle introduction to why Bayesian Optimization works, how the Ax library makes it simple, and a step-by-step walkthrough of this Hartmann demo—including full code and mathematical background—check out that post first.



This function has multiple local optima and one global optimum, making it a perfect test case. A grid search with a step of 0.1 would require $10^6$ evaluations. Let's see how Ax handles it:

```python
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig

# Create the Ax Client
client = Client()

# Define the search space: 6 variables, each between 0 and 1
parameters = [
    RangeParameterConfig(name=f"x{i}", parameter_type="float", bounds=(0, 1))
    for i in range(1, 7)
]

client.configure_experiment(parameters=parameters)
client.configure_optimization(objective="-hartmann")  # Minimize
```

The optimization loop is remarkably simple: ask Ax for suggested trials, run them, and report results:

```python
for _ in range(10):
    # Ask Ax for the next batch of parameters to try
    trials = client.get_next_trials(max_trials=5)

    for trial_index, parameters in trials.items():
        # Evaluate the function (in a real scenario, this is your experiment)
        result = hartmann6(**parameters)

        # Report the result back to Ax
        client.complete_trial(
            trial_index=trial_index,
            raw_data={"hartmann": result}
        )

best_parameters, prediction, _, _ = client.get_best_parameterization()
```

**Result:** Ax found a value of **-3.30** in just **45 trials**, while the known global optimum is **-3.32**. That's a 0.9% difference, found with a fraction of the effort a grid search would require.

### Step 2: Optimizing the Marketing Campaign

Now let's apply Ax to a real-world scenario. We're putting ourselves in the shoes of a **DSP (Demand Side Platform)** --- the system that decides, on behalf of an advertiser, how much to bid for each ad impression in real time.

The DSP receives thousands of auction requests per second. For each one, it must decide: **Should I bid? And if so, how much?** Bid too high, and you burn through the advertiser's budget on low-quality impressions. Bid too low, and you lose every auction and show zero ads. The bidding strategy formula we defined earlier controls this behavior, but its hyperparameters (`base_bid`, `ctr_coefficient`, etc.) need to be tuned to find the sweet spot.

In a live environment, you'd tune these parameters by running actual campaigns, each costing real money. Instead, we **simulate** this process using the [iPinYou dataset](https://contest.ipinyou.com/), a real-world RTB dataset from a Chinese DSP containing individual auction requests, bid prices, impressions, and clicks for a week of campaign data.

If a bid is won, the DSP will pay the price of the impression. Then, based on the real impression data, the DSP will track is the ad was clicked or not.

For each day of data, the simulation replays every auction request as if the DSP were bidding in real time:
1. **Predicts the CTR** for the incoming impression using an XGBoost model trained on prior days
2. **Calculates the bid amount** using our bidding strategy formula and the current set of hyperparameters
3. **Determines if the bid won** the auction (by comparing against the actual floor price in the dataset)
4. **Tracks the outcome** budget consumed, impressions won, and clicks received

Each simulation run with a different set of hyperparameters is equivalent to running an entire day-long campaign. The question Ax answers is: **which combination of hyperparameters produces the best campaign performance?**

Here's how we set up Ax to optimize the campaign parameters:

```python
client = Client()

parameters = [
    RangeParameterConfig(name="base_bid", parameter_type="float", bounds=(20, 500)),
    RangeParameterConfig(name="bid_to_pay_ratio", parameter_type="float", bounds=(0, 1)),
    RangeParameterConfig(name="ctr_coefficient", parameter_type="float", bounds=(0, 1)),
    RangeParameterConfig(name="pacing_coefficient", parameter_type="float", bounds=(0, 1)),
]

client.configure_experiment(parameters=parameters)
client.configure_optimization(objective="clicks")
```

Each trial runs the simulation for one day with a different set of parameters, and the results are fed back to Ax:

```python
for date in simulation_dates:
    trials = client.get_next_trials(max_trials=1)
    trial_index = list(trials.keys())[0]
    params = trials[trial_index]

    # Run the DSP simulation with the suggested parameters
    _, _, _, budget_spent, clicks = dsp_simulation(
        date, params["base_bid"], budget,
        params["bid_to_pay_ratio"],
        params["ctr_coefficient"],
        params["pacing_coefficient"]
    )

    # Report results back to Ax
    client.complete_trial(
        trial_index=trial_index,
        raw_data={
            "clicks": clicks,
            "impressions": impressions,
            "budget_spent": budget_spent
        }
    )
```

To illustrate the difference, a manual experiment with fixed parameters (`base_bid=100, coefficients=0`) yielded **83 clicks**. With Ax-optimized parameters, the simulation achieved up to **103 clicks** with the same budget --- a **24% improvement** just by intelligently tuning the hyperparameters.

### Step 3: Multi-Objective Optimization

In practice, advertisers don't just want to maximize clicks --- they also want to **minimize cost**. These two objectives are in tension: spending more generally yields more clicks, but with diminishing returns.

Ax supports multi-objective optimization natively, producing a **Pareto frontier** that shows the optimal trade-offs:

```python
client.configure_optimization(
    objective="-budget_spent, clicks",
    outcome_constraints=[
        "budget_spent <= 6009888.0",
        "clicks >= 1"
    ]
)
```

After running the experiment, we extract the Pareto frontier:

```python
frontier = client.get_pareto_frontier()

for parameters, metrics, trial_index, arm_name in frontier:
    print(f"Trial {trial_index}: {metrics}")
```

The Pareto frontier gives the advertiser a set of optimal configurations to choose from. For example:
- **Low budget** (~640K): 37 clicks --- efficient but limited reach
- **Medium budget** (~2.1M): 73 clicks --- balanced approach
- **High budget** (~5M): 127 clicks --- maximum reach but higher cost per click

![Pareto Frontier](images/rtb-clicks-pareto-frontier.png)

This is powerful because it transforms a vague question ("What's the best bid strategy?") into a concrete set of trade-offs the advertiser can reason about.

## Wrapping Up: Why Ax Deserves a Spot in Your Toolkit

Ax turns the complex mathematics of Bayesian Optimization into a practical, accessible tool. Here's what makes it compelling:

1. **Efficiency**: Find near-optimal solutions in a fraction of the trials that grid search or random search would require
2. **Multi-objective support**: Real-world problems rarely have a single objective, Ax handles this natively
3. **Simplicity**: The API is intuitive enough that you can start optimizing in minutes, while BoTorch handles the heavy lifting underneath
4. **Constraints**: You can easily set real-life limits for the optimization, like a maximum budget or a minimum number of clicks

### Other Applications

Ax and Bayesian Optimization are applicable wherever you have expensive black-box experiments:

- **Hyperparameter tuning** for ML models (learning rate, architecture choices, regularization)
- **A/B testing and multi-armed bandits** for content recommendation and feature experimentation
- **Materials science** Meta used Ax to develop [lower-carbon concrete formulations](https://engineering.fb.com/2025/07/16/data-center-engineering/ai-make-lower-carbon-faster-curing-concrete/)
- **Manufacturing** tuning process parameters for quality and yield

### Other Libraries Worth Exploring

If Bayesian Optimization interests you, here are some related tools:

- **[BoTorch](https://botorch.org/)**: The lower-level library behind Ax, for custom acquisition functions and advanced GP models
- **[Optuna](https://optuna.org/)**: Lightweight alternative with Tree-structured Parzen Estimator (TPE) optimization

---

## References

- [Ax: Adaptive Experimentation Platform](https://ax.dev/)
- [BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization](https://arxiv.org/abs/1910.06403)
- [Ax: A Platform for Adaptive Experimentation (Paper)](https://openreview.net/forum?id=U1f6wHtG1g)
- [Engineering At Meta: Efficient Optimization With Ax](https://engineering.fb.com/2025/11/18/open-source/efficient-optimization-ax-open-platform-adaptive-experimentation/)
- [Engineering At Meta: Using AI to make lower-carbon concrete](https://engineering.fb.com/2025/07/16/data-center-engineering/ai-make-lower-carbon-faster-curing-concrete/)
- [iPinYou Global RTB Bidding Algorithm Competition Dataset](https://contest.ipinyou.com/)
- [Real-Time Bidding Benchmarking with iPinYou Dataset (Zhang, Yuan, Wang)](https://github.com/wnzhang/make-ipinyou-data)
- [Google Vizier: A Service for Black-Box Optimization (2017)](https://dl.acm.org/doi/10.1145/3097983.3098043)
- [Introduction to Bayesian Optimization (blog post)](blog_introduction_to_bayesian_optimization.md)
