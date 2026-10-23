# Causal Analysis of Agent Skill and Luck

- This folder has 2 simulations that separate skill from luck, with the setup
  to run them in a container:
  - `causal_success_*`: agents with different talents meet lucky and unlucky
    events, and their capital compounds over time
  - `agent_benchmark_*`: AI agent configs are compared on a benchmark, and the
    best observed score is too high on average (the winner's curse)
- `skills_usage.md` records how the `.claude` skills were applied to the code

## Quick Start

- From the root of the repository, change your directory to this folder:
  ```bash
  > cd research/Causal_Analysis_of_Agent_Skill_And_Luck
  ```

- Build the image:
  ```bash
  > ./docker_build.sh
  ```

- Run the container and launch Jupyter:
  ```bash
  > ./docker_jupyter.sh
  ```

- Open the notebooks in this order:
  1. `causal_success_API.ipynb`: the building blocks, i.e., the `Agent`
     class, the simulation, the inequality metrics, the policies, and the
     Bayesian model
  2. `causal_success_example.ipynb`: the full analysis, i.e., the
     simulation, double machine learning, causal forests, and the policy
     comparison
  3. `agent_benchmark_API.ipynb`: one small benchmark, step by step
  4. `agent_benchmark_example.ipynb`: the full sweep of the winner's curse

- All seeds are fixed, so every run gives the same results

- For more information on the Docker build system refer to [Project template
  readme](https://github.com/gpsaggese/umd_classes/blob/master/class_project/project_template/README.md).

## Notation

- The code and the notebooks use the notation of the white paper
  - The paper's symbol is used in the markdown, and the code name in the code

### Skill-luck decomposition

- From the paper's "Mathematical Framework" section

| Symbol | Meaning | In the code |
| :----- | :------ | :---------- |
| $A$ | An agent | One `Agent`, or one benchmark config `k` |
| $I_A$ | Information set of agent $A$ | Not modeled |
| $S_A$ | Strategy of agent $A$ | The talents of an `Agent`, or the skill `theta` of a config |
| $Y$ | Outcome | `capital` at the end of the run, or `resolved` on one task |
| $\bar{Y}_A = E[Y \mid S_A, I_A]$ | Expected outcome, i.e., skill | `ybar` in `agent_benchmark_utils` |
| $L_A(Y) = Y - \bar{Y}_A$ | Luck component of the outcome | The score minus `ybar` in the agent benchmark |
| $L_A(E)$ | Luck of one event $E$ | Not computed in the code |
| $U_A(E)$ | Utility of event $E$ | Not computed in the code |
| $S(E \mid I_A) = -\log P(E \mid I_A)$ | Surprise of event $E$ | Not computed in the code |
| $C_A(E)$ | Control of agent $A$ over event $E$ | Not computed in the code |
| $B_i$, $A_i$, $S_i$ | Circumstances, choices, and shocks ("Societal Modeling" section) | Not used |

- Always write the argument of $L_A$: $L_A(Y)$ is the luck of an outcome and
  $L_A(E)$ the luck of an event
- $S_A$ (strategy), $S(E \mid I_A)$ (surprise), and $S_i$ (shocks) are
  different symbols

### Causal success simulation

- From the paper's "Simulation Design" subsection

| Symbol | Meaning | In the code |
| :----- | :------ | :---------- |
| $i$, $N$ | Agent index and number of agents | `Agent.id`, `n_agents` |
| $t$, $T$ | Period index and number of periods | `n_periods` |
| $t_i^{(1)}$ | Intensity | `talent["intensity"]` |
| $t_i^{(2)}$ | IQ, i.e., the probability $p_i$ of using a lucky event | `talent["iq"]` |
| $t_i^{(3)}$ | Networking | `talent["networking"]` |
| $t_i^{(4)}$ | Initial capital | `talent["initial_capital"]` |
| $\mathbf{T}_i$ | Talent vector | `talent`, and its norm `talent_norm` |
| $q_i = \sigma(\alpha(t_i^{(1)} - 0.5))$ | Event exposure probability | `get_event_probability()`, with `alpha = 2.0` |
| $C_{i,t}$ | Capital of agent $i$ at period $t$ | `capital_history[t]`, and `capital` at the end |
| $\Delta_i^{\text{lucky}}$, $\Delta_i^{\text{unlucky}}$ | Impact of one event | `lucky_mean`, `lucky_std`, `unlucky_mean`, `unlucky_std` |

- The code says "capital" and not $C$, since $C_A(E)$ is the control in the
  skill-luck decomposition
- In the Bayesian model of `fit_bayesian_luck_model()`, `alpha` is the
  intercept and `sigma` the residual standard deviation, not the $\alpha$ and
  $\sigma$ of $q_i$

### Agent benchmark simulation

| Symbol | Meaning | In the code |
| :----- | :------ | :---------- |
| $K$, $N$, $S$ | Configs, tasks, and seeds | `n_configs`, `n_tasks`, `n_seeds` |
| $\theta_k$ | Skill of config $k$ | `theta` |
| $b_i$ | Difficulty of task $i$ | `b` |
| $u_{ki}$ | Interaction of config $k$ with task $i$ | `u` |

- $\theta_k$, $b_i$, and $u_{ki}$ are parameters of this simulation and are
  not symbols of the paper
- Here $S$ is the number of seeds and $N$ the number of tasks, as in
  `SimParams`
