# A Causal Analysis of Success in Modern Society

**Author:** Krishna Kishore Buddi  
**Course:** MSML610 – Modern Statistical Machine Learning  
**Instructor:** Prof. GP Saggese

## 1. Abstract

Why do measures of success—such as wealth, academic influence, or professional
recognition—exhibit extreme inequality, while underlying human abilities appear
broadly similar? This project explores that question using an agent-based
simulation combined with modern causal inference techniques.

We model success as a **multiplicative stochastic process** shaped by both
talent and random events, then apply **Double Machine Learning (DML)** and
**Causal Forests** to estimate the causal effect of "luck" while controlling for
talent. A small **Bayesian regression** extension provides posterior uncertainty
for the estimated luck effect. Across runs, the analysis shows that stochastic
factors exert a much stronger influence on long-run outcomes than modest talent
differences alone. These findings challenge simple meritocratic narratives and
inform policy design and opportunity allocation.

## 2. Motivation

Human traits such as intelligence, effort, and creativity tend to cluster around
average values, often resembling **normal distributions**. In contrast, outcomes
such as income, citations, and firm size follow **highly skewed, heavy-tailed
distributions**, where a small minority accumulates a disproportionate share of
total "success."

This mismatch suggests that success is shaped by more than just individual
merit. Understanding how randomness interacts with **compounding dynamics**
helps explain why inequality emerges so reliably, even when starting conditions
are similar and talent differences are relatively modest.

## 3. Research Question and Hypothesis

**Core question**

> Why does success follow a power-law–like distribution when human talent
> appears roughly normally distributed?

**Hypothesis**

Random **multiplicative events** play a larger _causal_ role in determining
long-term success than talent differentials alone. Because gains and losses
compound over time, small stochastic differences—especially early in life—can
generate large divergences in outcomes.

## 4. Model Overview

### 4.1 Agents

We simulate a population of **100 agents**. Each agent (i) has four attributes
in [0, 1]:

- **Intensity** – sustained effort and activity level.  
  Higher intensity → higher probability of encountering events (good and bad).

- **IQ** – ability to capitalize on beneficial opportunities.  
  IQ does not create opportunities, but gates whether an agent can convert a
  good event into a gain.

- **Networking** – social connectivity.  
  Modulates the chance that a good event "spills over" to another agent,
  representing referrals and informal networks.

- **Initial Capital** – starting resources.  
  Set to 1.0 for all agents in the baseline, so inequality arises _endogenously_
  from dynamics rather than initial wealth.

All talent attributes are drawn from N(0.5, 0.15) and **clipped to [0, 1]** to
avoid unrealistic extremes.

### 4.2 Event Process and Dynamics

The simulation runs for **80 time periods**. In each period:

1. A fixed number of **beneficial ("lucky")** and **detrimental ("unlucky")**
   events is generated.
2. Events are assigned probabilistically; exposure probability is a sigmoid
   function of intensity.
3. When a beneficial event hits an agent:
   - With probability equal to the agent's IQ, it is successfully exploited.
   - Capital is updated multiplicatively: C\_{t+1} = C_t(1 + Δ_lucky)
4. When a detrimental event hits an agent:
   - Capital is updated multiplicatively: C\_{t+1} = C_t(1 - Δ_unlucky)
   - Capital is floored at a small positive value (e.g., 0.01) to avoid collapse
     to exactly zero.
5. **Networking spillovers**: a fraction of lucky events generate secondary,
   smaller boosts for other agents, allocated in proportion to networking
   scores.

Event magnitudes are drawn from normal distributions and clipped to realistic
ranges (e.g., lucky impacts ~25% ± 8%, unlucky impacts ~15% ± 5%).

## 5. Analysis Pipeline

The analysis is organized into the following steps:

1. **Descriptive statistics and inequality measures**
   - Distribution of final capital
   - Gini coefficient
   - Top/bottom wealth shares
   - Correlations between talent, luck, and log(capital)

2. **Top performer inspection**
   - Compare top-decile agents to the population in terms of talent and
     experienced lucky events.
   - Typical finding: top agents are not always the most "talented," but they
     tend to have many more lucky events.

3. **Double Machine Learning (DML)**
   - Outcome: log(final capital)
   - Treatment: number of lucky events
   - Controls: talent vector (intensity, IQ, networking)
   - Use DML to estimate the **causal effect** of lucky events while controlling
     flexibly for confounding.
   - Robust pattern: each additional lucky event increases final capital by
     roughly **10–15%**, conditional on talent.

4. **Causal Forests (heterogeneous treatment effects)**
   - Use Causal Forests to estimate **Conditional Average Treatment Effects
     (CATEs)**.
   - Examine how the effect of luck varies with agent features (e.g., IQ,
     networking).
   - This informs which agents benefit most from an additional opportunity.

5. **Bayesian regression (optional extension)**
   - Fit a Bayesian regression model: log(C_final) ~ alpha + beta_luck _
     lucky_events + beta_int _ intensity + beta_iq _ iq + beta_net _
     networking + epsilon
   - Implemented with **PyMC** and summarized via **ArviZ**.
   - Provides a posterior distribution for the luck effect (beta_luck) with
     credible intervals.

## 6. Policy Experiments

The project also compares different one-shot **resource allocation policies**
using the simulation as a sandbox.

### Implemented Policies (`run_policy_simulation`)

- **Egalitarian**  
  Equal share of a fixed budget for every agent.

- **Meritocratic**  
  Resources allocated in proportion to a talent summary (e.g., norm of the
  talent vector).

- **Performance-based**  
  Allocation proportional to current capital (rich-get-richer).

- **Random**  
  A random agent receives the full budget.

- **CATE-optimal**  
  Resources allocated proportionally to **non-negative CATE estimates** from the
  Causal Forest. Implemented via `policy="cate_optimal"` in
  `run_policy_simulation`, with `cate_values` passed as an argument.

### Qualitative Findings

Across runs, we typically see:

- **Egalitarian** reduces inequality but may slightly reduce total output
  compared to targeted schemes.
- **Performance-based** amplifies inequality substantially and can concentrate
  gains in already advantaged agents.
- **Meritocratic** often lies between egalitarian and performance-based in both
  inequality and efficiency.
- **Random** can perform surprisingly well as a baseline.
- **CATE-optimal** tends to maximize total output by directing resources to
  agents who are estimated to benefit the most from an extra opportunity.

These trade-offs illustrate how policy design interacts with multiplicative
dynamics and randomness.

## 7. Repository Structure

Key files in the project:

- **Core utilities**
  - `causal_success_utils.py` Main toolbox:
    - Agent class and simulation engine
    - Inequality metrics and summary statistics
    - `run_policy_simulation` with multiple allocation rules (including
      `cate_optimal`)
    - Bayesian regression helpers (PyMC + ArviZ)

- **API documentation**
  - `causal_success.API.md` Textual API specification: functions, arguments, and
    example usage.
  - `causal_success.API.ipynb` Notebook version of the API walkthrough with
    runnable examples.

- **Example analysis**
  - `causal_success.example.md` Narrative, end-to-end example: simulation → DML
    → Causal Forest → policy comparison.
  - `causal_success.example.ipynb` The corresponding notebook with all code and
    plots.

- **Tutorial / theory**
  - `causal_success_tutorial.md` Higher-level explanation of the modeling
    choices, theory, and connections to real-world inequality.

- **Environment / infrastructure**
  - `requirements.txt` Python dependencies (NumPy, pandas, scikit-learn, econml,
    pymc, arviz, etc.).
  - `Dockerfile` Reproducible environment for running the notebooks.
  - `docker_build.sh`, `docker_jupyter.sh`, `run_jupyter.sh` Convenience scripts
    for building the image and launching Jupyter Lab inside the container.

- **Top-level**
  - `README.md` (this file)

## 8. Setup and Reproducibility

The project targets **Python 3.10** and is designed to be fully reproducible
with **Docker**. You can either use Docker (recommended for clean grading) or a
local Python environment.

### 8.1 Using Docker (Recommended)

1. **Build the image**

   From the project root:

   ```bash
   ./docker_build.sh
   ```

2. **Launch Jupyter Lab inside the container**

   Depending on your scripts, use either:

   ```bash
   ./docker_jupyter.sh
   ```

   or

   ```bash
   ./run_jupyter.sh
   ```

3. **Open the notebook UI**
   - Copy the Jupyter URL printed in the terminal (usually
     `http://127.0.0.1:8888/...`)
   - Open it in your browser.
   - Navigate to `causal_success.API.ipynb` or `causal_success.example.ipynb`.

All code, including simulations, causal models, and policy experiments, can be
run end-to-end from within the container.

### 8.2 Local Python Environment (Optional)

If you prefer not to use Docker:

1. Create and activate a virtual environment (example with `venv`):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scriptsctivate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start Jupyter Lab:

   ```bash
   jupyter lab
   ```

4. Open the notebooks as above.

## 9. How to Use the Project

A typical workflow:

1. **Explore the API**
   - Open `causal_success.API.ipynb`.
   - Inspect how to:
     - Create a population,
     - Run the simulation,
     - Compute inequality,
     - Use `run_policy_simulation`,
     - Run the Bayesian model.

2. **Run the main analysis**
   - Open `causal_success.example.ipynb`.
   - Execute cells in order:
     - Simulate a baseline run,
     - Examine inequality,
     - Run DML and Causal Forest,
     - Generate CATE-optimal allocations,
     - Compare policy outcomes.

3. **Dive into theory**
   - Read `causal_success_tutorial.md` for the narrative and theoretical
     context.

## 10. Limitations and Extensions

This is a **stylized model** intended to isolate a core mechanism:

- It does **not** explicitly model structural barriers, institutions, or
  strategic behavior.
- Talent is static; in reality, success and failure feed back into skills and
  opportunities.
- Opportunity structures and shocks are simplified.

Natural extensions include:

- Endogenous talent evolution over time.
- Richer network structures and community effects.
- More realistic opportunity processes calibrated to empirical data.
- Policy experiments with multi-period interventions and constraints.
