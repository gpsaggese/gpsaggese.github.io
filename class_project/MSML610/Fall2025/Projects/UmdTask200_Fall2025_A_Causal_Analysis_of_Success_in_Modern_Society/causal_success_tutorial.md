# Causal Analysis of Success: Tutorial Documentation

## Overview

This project looks at a simple but uncomfortable question: if most people’s abilities are “kind of average” (roughly normally distributed), why do real-world outcomes look so unequal? Wealth, citations, and company sizes often follow *power-law* patterns, where a small group ends up with a huge share of the total.

To explore this, we build an agent-based simulation where people start from equal conditions, experience good and bad events over time, and grow (or shrink) through **multiplicative** dynamics. Then we layer in causal inference tools to separate correlation from causation—specifically, Double Machine Learning (DML) and Causal Forests.

The goal is not to perfectly model society. The goal is to understand the *mechanism* that can create extreme inequality even when talent differences are modest.

---

## The core idea in plain terms

### The central question

Imagine two people who start at the same time, with roughly similar ability and motivation. Ten years later, one is massively successful and the other is doing “fine.” What happened?

One possibility is that talent differences alone explain it. Another is that random events—timing, exposure, opportunity, and setbacks—compound over time. This project explores the second explanation, without pretending talent doesn’t matter.

### Why this matters

This pattern shows up everywhere:

- **Wealth**: a small fraction of people control a large share of resources.
- **Academic impact**: most papers get a few citations, while a few become extremely influential.
- **Companies**: most are small, while a handful become massive.

Meanwhile, core human traits like intelligence, persistence, and creativity tend to cluster around a middle range. That mismatch is a big hint: outcomes are being shaped by something besides raw ability.

---

## Model components

### What an “agent” represents

Each simulated person is an *agent* with four traits (all scaled between 0 and 1), drawn from a normal distribution and then clipped into range:

- **Intensity**: how active the agent is (effort, persistence, seeking opportunities).  
  In the model this controls “surface area”: higher intensity → more exposure to events.

- **IQ**: how likely an agent is to successfully take advantage of a good opportunity when it appears.  
  This doesn’t *create* opportunities; it affects whether the agent can *convert* them into gains.

- **Networking**: how likely good opportunities can spill over through connections.  
  This is a simplified way to represent referrals, social ties, and access.

- **Initial Capital**: set to \$1 for everyone in the base simulation.  
  This is intentional: we want inequality to arise from compounding and randomness, not inherited advantages.

---

## Event mechanics

### The key feature: multiplicative effects

Each time period includes both good and bad events. Events matter because they change capital by a **percentage**, not a fixed amount:

- Good event: capital becomes `capital × (1 + impact)`
- Bad event: capital becomes `capital × (1 – impact)`

This is why compounding happens. A 20% gain means much more if you already have a lot of capital.

A quick example:

- \$100 → +20% → \$120  
- \$1,000 → +20% → \$1,200  

Same percentage change, but the gap grows faster for those already ahead.

### How event impacts are generated

Impact magnitudes are sampled from normal distributions and then clipped to keep them realistic:

- Good events: mean 25%, std 8%, clipped to 5%–50%
- Bad events: mean 15%, std 5%, clipped to 5%–30%

Events are not assigned uniformly. Agents with higher intensity have higher exposure probabilities, meaning they encounter *more opportunities and more risks*.

---

## Simulation dynamics

### Population creation

We generate **100 agents**. Each talent dimension is drawn from:

- mean = 0.5
- std = 0.15
- clipped to [0, 1]

This produces a realistic distribution where most people sit near the center with fewer extremes.

### Time evolution

The simulation runs for **80 periods**. Each period:

1. Generates a fixed set of events (default: 5 lucky, 5 unlucky)
2. Assigns events probabilistically based on intensity (exposure)
3. Applies multiplicative impacts
4. Records capital history

### What tends to emerge

Even though everyone starts equal, the results typically become highly unequal:

- Capital becomes right-skewed (a long tail of big winners)
- Gini often lands in the 0.35–0.50 range
- The top 10% hold a large share of wealth
- Capital spans a wide range despite equal starting points

This is the main “takeaway pattern”: **compounding + randomness = heavy inequality**.

---

## Analysis methods

### 1) Correlation: talent vs luck

A simple starting point is correlation:

- Talent norm vs log(capital): usually weak (around 0.05–0.15)
- Lucky events vs log(capital): usually strong (around 0.70–0.80)

This often produces a “luck dominates” ratio in the 10-to-1 range.

But correlation alone isn’t causal, because talent can influence both:
- event exposure (intensity)
- capitalization ability (IQ)

So we need causal tools.

### 2) Top performer analysis

A practical way to build intuition is to inspect the winners. In many runs, top performers:

- are not the most talented on average
- experienced meaningfully more lucky events than the population mean

The story becomes: “being reasonably capable helps, but being unusually lucky matters more.”

### 3) Double Machine Learning (DML)

DML is used to estimate the causal effect of luck while controlling for confounding.

Roughly:

1. Predict **treatment** (lucky events) from talent
2. Predict **outcome** (log final capital) from talent
3. Remove the “talent explained” portions (residualize)
4. Relate residual treatment to residual outcome

This gives a cleaner estimate of “what one more lucky event causes,” holding talent constant.

A common result is that **each additional lucky event produces about 10–15% higher final capital**, even after controlling for talent.

### 4) Causal Forests

DML gives an average effect. Causal Forests go further: they estimate **heterogeneous effects** (CATE). That tells us whether some types of agents benefit more from luck than others.

Typical patterns:
- Mean CATE roughly matches DML
- There’s real variation (std ~0.02–0.04)
- Higher IQ/networking often predict higher treatment effects

This is useful for policy thinking: if you can expand opportunities, *who benefits most*?

### 5) Optional: Bayesian regression

In addition to DML and Causal Forests, the codebase includes a simple Bayesian regression on log(final capital). The model regresses log-capital on the number of lucky events and the talent dimensions (intensity, IQ, networking). This gives us a full posterior distribution over the “luck effect” and produces credible intervals for how strongly lucky events influence outcomes, providing a complementary robustness check to the DML estimate.


---

## Policy evaluation

### Allocation strategies

We compare multiple ways to distribute a fixed resource budget:

- **Egalitarian**: everyone gets the same amount
- **Meritocratic**: proportional to talent
- **Performance-based**: proportional to current capital (reinforces inequality)
- **Random**: lottery allocation

### What tends to happen

- Egalitarian tends to reduce inequality (equity)
- Performance-based usually worsens inequality dramatically
- Random is surprisingly competitive in some settings
- Meritocratic often sits in the middle

There’s no “one best” policy—there are trade-offs.

---

## Key insights

### For understanding success

- **Compounding dominates.** Percentage-based growth naturally stretches distributions.
- **Timing matters.** Early events matter more than late events due to compounding.
- **Talent is not irrelevant**, but it’s often not the main driver among already-capable people.

### For individuals

- Increasing exposure (activity) increases opportunity encounters.
- Luck is real; recognizing it builds humility and resilience.
- Buffers help: one bad event can permanently derail trajectories without protection.

### For policymakers

- Broad access to opportunity can outperform narrow targeting.
- Early interventions compound into larger long-run differences.
- Extreme inequality can waste talent by limiting who gets to convert ability into outcomes.

---

## Extensions and limitations

### Extensions worth exploring

- Talent evolution (success builds skill; failure erodes confidence/ability)
- Explicit networks (graph structures instead of an abstract networking score)
- More event types (e.g., rare “big” opportunities vs frequent small ones)
- Calibration to real distributions (so results match observed wealth/citation patterns)
- CATE-optimal targeting : use estimated treatment effects from the Causal Forest to target resources to agents who benefit most. A simple version is implemented via the cate_optimal policy in run_policy_simulation, using CATE estimates from the Causal Forest; this can be extended further.


### Limitations (what this is *not*)

This is a stylized model. It intentionally simplifies:
- institutional barriers and structural inequality
- strategic decision-making
- many real-world feedback loops

The purpose is to isolate a core mechanism and show how far it can go on its own.

---

## Implementation notes

### Requirements

The project targets Python 3.10 and uses common scientific libraries plus EconML for DML and Causal Forests.

### How to run

1. Build the Docker image:
   ```bash
   ./docker_build.sh
