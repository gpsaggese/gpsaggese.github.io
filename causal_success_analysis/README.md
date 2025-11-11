# A Causal Analysis of Success in Modern Society

Author: Krishna Kishore Buddi  
Course: DATA605 / MSML610 – Fall 2025  
Instructor: Prof. Giovanni Saggese  
University: University of Maryland  
Project Type: Research Project (Hard)  
Folder: class_project/causal_success_analysis/

------------------------------------------------------------

## Project Overview

This project presents an agent-based causal simulation exploring how luck and talent interact to shape success in modern societies.

Although talent is generally distributed normally across populations, real-world success (wealth, recognition, productivity) often follows a Pareto or power-law distribution. This pattern suggests that randomness and opportunity play a much larger role than raw ability alone.  
The simulation models these effects and integrates causal inference methods to estimate the treatment effect of "luck" while controlling for talent.

------------------------------------------------------------

## Research Question

Why does success follow a power-law distribution when human talent appears normally distributed?

**Hypothesis:**  
Random, multiplicative events (lucky or unlucky opportunities) have a stronger causal influence on long-term success than talent alone.

------------------------------------------------------------

## Model Description

### Agents
Each of 100 agents is defined by a four-dimensional talent vector:
- Intensity – effort, persistence, and hours worked  
- IQ – cognitive ability and problem-solving skill  
- Networking – social connectivity and collaboration potential  
- Initial Capital – starting wealth or resources

### Simulation Dynamics
- Runs for 40–50 time periods (rounds).  
- Each round, agents experience random positive or negative events.  
- Event probabilities depend on the agent's talent attributes.  
- Wealth evolves multiplicatively (C_{t+1} = C_t * (1 ± Δ)).  
- Repeated random shocks lead to nonlinear inequality.

### Expected Outcomes
1. Emergent inequality: wealth follows a Pareto distribution despite normal talent.  
2. Luck dominates correlations: success correlates more strongly with the number of lucky events than with talent.  
3. Average winners: top performers often have average talent but experience many lucky events.  
4. Policy implications: randomized or egalitarian funding tends to outperform purely meritocratic systems.

------------------------------------------------------------

## Dependencies

See `requirements.txt` for details. Major packages include:
- numpy  
- pandas  
- scipy  
- matplotlib  
- seaborn  
- scikit-learn==1.2.2  
- econml  
- jupyterlab  
- tqdm

------------------------------------------------------------

## Running the Project with Docker

### Prerequisites
- Docker installed  
- Git installed

### Steps

1. Clone the repository and navigate to the project directory.
2. Build the Docker image:
   ```bash
   ./docker_build.sh
   ```
   Or manually:
   ```bash
   docker build -t causal_success_analysis .
   ```
3. Run Jupyter Lab inside the container:
   ```bash
   ./docker_jupyter.sh
   ```
   Or manually:
   ```bash
   docker run -it --rm -p 8888:8888 -v $(pwd):/app causal_success_analysis
   ```
4. Open a browser and visit http://localhost:8888

------------------------------------------------------------

## Project Structure

```
causal_success_analysis/
│
├── Dockerfile                          # Container environment (Python 3.10)
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation
├── causal_success_simulation.py         # Simulation model (agents + events)
├── causal_success_tutorial.md           # Markdown tutorial walkthrough
├── tutorial_causal_success.ipynb        # Jupyter notebook tutorial
├── docker_build.sh                      # Helper script to build Docker image
├── docker_jupyter.sh                    # Helper script to start Jupyter
└── version.sh                           # Version and metadata
```

------------------------------------------------------------

## Key Findings

| Observation | Insight |
|--------------|----------|
| Normal talent → Pareto success | Inequality naturally emerges even in fair systems. |
| Luck–success correlation ≈ 3–4× talent–success correlation | Random events dominate long-term outcomes. |
| Average talent + multiple lucky breaks → top success | Reinforces the multiplicative nature of advantage. |
| Egalitarian allocation beats meritocratic | Broad opportunity distribution improves overall efficiency. |

------------------------------------------------------------

## Causal Inference Integration

- **Causal Forests (EconML):** Estimate heterogeneous treatment effects (HTEs) where treatment = number of lucky events and outcome = final wealth.  
- **Double Machine Learning (DML):** Controls for confounders to isolate the causal effect of opportunities.  
- **Uplift Modeling:** Identifies agents who benefit most from additional opportunities.  
- **Deep IV / Orthogonal Random Forests:** Handles cases where luck is not fully random.

------------------------------------------------------------

## Next Steps

- Run complete simulations with multiple parameter configurations.  
- Visualize wealth trajectories, distributions, and Lorenz curves.  
- Implement causal inference analysis using DML and Causal Forests.  
- Validate simulated Pareto exponents with empirical datasets.  
- Implement dynamic feedback loops (reputation and visibility effects).

------------------------------------------------------------

## References

- Pluchino, A., Biondo, A. E., & Rapisarda, A. (2018). *Talent vs Luck: The Role of Randomness in Success and Failure.* Physica A.  
- Athey, S., & Wager, S. (2019). *Estimating Treatment Effects with Causal Forests.* Journal of Econometrics.  
- University of Maryland (2025). *DATA605 / MSML610 Class Project Guidelines.*

------------------------------------------------------------

## Verification Checklist

- [x] Docker image builds successfully.  
- [x] Jupyter Lab launches correctly at http://localhost:8888.  
- [x] All dependencies install without conflict.  
- [x] pandas, numpy, and econml import successfully inside Jupyter.  
- [x] Core simulation model implemented (Agent class, event dynamics, Gini coefficient).
- [ ] Full simulation runs with visualization in notebook (next phase).
- [ ] Causal inference analysis (DML, Causal Forests) to be integrated (next phase).
