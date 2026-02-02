# Visual Script for Multi-Armed Bandits

## Cell 1: Introduction - Casino Slot Machines

- Type: Interactive
- Visualization:
  - Draw 3 "slot machines" generating a random number from -1 to 1
  - Each machine displays a question mark on its screen or a value
- Interactive widget:
  - Choose a "slot machine"
  - Toggle "Show True Means" (on/off)
  - Slider for true mean of each machine mu_1, mu_2, ..., mu_K (0-1)
  - Reset "total winnings" and "number of coins"
- Display:
  - Bar chart showing true mean of each machine when revealed
  - Total winnings
  - Counter showing budget of coins remaining
- Comment box: "You have 10 coins. Each machine has a different unknown payout
  rate. How do you maximize your winnings?"

## Cell 2: Exploration vs Exploitation Dilemma

- Type: Interactive
- Use the set up from cell1_casino_slot_machines
- Visualization:
  - There are three strategies
  - Left side: "Explore" - trying different machines randomly
  - Right side: "Exploit" - sticking with best known machine
  - Balanced
- Interactive widget:
  - Seed widget
  - Number of coins
  - Slider for exploration probability epsilon (0-1)
  - Button "Run 100 Trials"
  - Display: Total reward for exploration-only, exploitation-only, and balanced
    strategy
- Display:
  - Three line plots showing cumulative reward over time
  - Highlight when each strategy succeeds/fails
- Comment box: "Pure exploration learns but earns little. Pure exploitation gets
  stuck on suboptimal choices. Balance is key."
- Purpose: Demonstrate the fundamental tradeoff visually

<start>
## Cell 3: Greedy Algorithm Failure
- Type: Interactive
- Visualization:
  - 3 slot machines with true means: mu_1=0.4, mu_2=0.7, mu_3=0.5
  - Timeline showing pulls and rewards
  - Highlight which machine is selected at each step
- Interactive widget:
  - Button "Run Greedy Algorithm"
  - Speed slider for animation
  - Reset button
- Display:
  - Show empirical mean estimates evolving over time
  - Color-code selected machine at each step
  - Display: "First pull got lucky on Machine 1 (reward=1), now stuck forever!"
- Comment box: "Greedy algorithm pulled Machine 1 first, got reward 1, and never
  tried the better Machine 2."
- Purpose: Show how greedy can get stuck on suboptimal arm
<end>

## Cell 5: Epsilon-Greedy Algorithm

- Type: Interactive
- Visualization:
  - Same 3 machines setup
  - Timeline showing pulls with two colors: green (exploit), blue (explore)
  - Empirical mean estimates displayed above each machine
- Interactive widget:
  - Slider for epsilon (0-0.5)
  - Button "Run Epsilon-Greedy"
  - Display: Number of times each arm pulled
  - Display: Cumulative reward
- Display:
  - Bar chart showing pull counts per machine
  - Line plot of cumulative reward over time
  - Highlight when exploration vs exploitation happens
- Comment box: "With epsilon=0.1, we explore 10% of time and exploit 90%. This
  prevents getting stuck."
- Purpose: Demonstrate epsilon-greedy balances exploration and exploitation

## Cell 6: Confidence Intervals for Each Arm

- Type: Interactive
- Visualization:
  - 3 machines with empirical mean and confidence intervals
  - Each machine shows: [lower_bound, empirical_mean, upper_bound]
  - Visualize as error bars or shaded regions
  - True mean shown as dotted line when revealed
- Interactive widget:
  - Slider for number of pulls per machine
  - Slider for confidence level (90%, 95%, 99%)
  - Display: Confidence intervals shrinking as more pulls happen
- Display:
  - Bar chart with error bars showing confidence intervals
  - Animation showing intervals shrinking with more data
- Comment box: "More pulls = smaller confidence intervals = less uncertainty."
- Purpose: Introduce confidence bounds and uncertainty quantification

## Cell 7: Upper Confidence Bound (UCB) Intuition

- Type: Interactive
- Visualization:
  - 3 machines showing empirical mean + exploration bonus
  - UCB index = empirical_mean + sqrt(2\*log(t)/N_i)
  - Display both components visually stacked
- Interactive widget:
  - Time slider t (1-100)
  - Display: UCB index for each machine
  - Highlight which machine has highest UCB at time t
- Display:
  - Stacked bar chart: empirical mean (blue) + exploration bonus (orange)
  - Arrow pointing to highest UCB
- Comment box: "UCB = Empirical Mean + Exploration Bonus. We pull the arm with
  highest UCB."
- Purpose: Show how UCB combines exploitation and exploration bonuses

## Cell 8: UCB Algorithm Simulation

- Type: Interactive
- Visualization:
  - 4 machines with true means: mu=[0.3, 0.5, 0.7, 0.4]
  - Timeline showing which arm pulled at each step
  - Empirical means and UCB values evolving
- Interactive widget:
  - Button "Run UCB Algorithm"
  - Speed slider
  - Display: Pull count for each arm over time
  - Display: Cumulative regret over time
- Display:
  - Line plot showing N_i(t) for each arm
  - Line plot showing cumulative regret L_T
  - Highlight optimal arm (i=3)
- Comment box: "UCB quickly identifies the best arm (Machine 3) and focuses on
  it while occasionally checking others."
- Purpose: Demonstrate UCB algorithm behavior and convergence to optimal arm

## Cell 9: UCB Exploration Bonus Decay

- Type: Interactive
- Visualization:
  - Single arm showing exploration bonus over time
  - Bonus = sqrt(2\*log(t)/N_i)
  - Graph showing how bonus decreases as N_i increases
- Interactive widget:
  - Slider for time t (1-1000)
  - Slider for pulls N_i (1-100)
  - Display: Bonus value
- Display:
  - 2D plot: x-axis = N_i, y-axis = exploration bonus
  - Show how bonus decays with 1/sqrt(N_i)
- Comment box: "The exploration bonus decreases as 1/sqrt(N_i). More pulls =
  less bonus = less exploration."
- Purpose: Visualize how exploration bonus shrinks with more data

## Cell 10: Regret Accumulation

- Type: Interactive
- Visualization:
  - Show optimal arm (green) vs chosen arm (red/yellow)
  - At each time step, show instantaneous regret = mu\* - mu_chosen
  - Cumulative regret accumulating over time
- Interactive widget:
  - Button "Run Algorithm" (choose: Random, Greedy, Epsilon-Greedy, UCB)
  - Display: Instantaneous regret at each step
  - Display: Cumulative regret curve
- Display:
  - Upper panel: Bar chart showing regret per step
  - Lower panel: Line plot showing cumulative regret L_T
- Comment box: "Regret = difference between optimal arm and chosen arm. UCB
  achieves logarithmic regret."
- Purpose: Visualize regret accumulation for different algorithms

## Cell 11: Comparing Algorithms - Regret Curves

- Type: Interactive
- Visualization:
  - Multiple regret curves on same plot
  - Algorithms: Random, Greedy, Epsilon-Greedy, UCB, Thompson Sampling
  - Log scale on y-axis to show logarithmic growth
- Interactive widget:
  - Checkboxes to select which algorithms to compare
  - Slider for time horizon T (100-10000)
  - Slider for number of arms K (2-10)
  - Button "Run Comparison"
- Display:
  - Line plot with multiple curves
  - Legend showing algorithm names
  - Highlight: UCB and Thompson Sampling have O(log T) regret
- Comment box: "Random: O(T), Greedy: O(T), Epsilon-Greedy: O(T^(2/3)), UCB:
  O(log T)."
- Purpose: Compare regret growth rates of different algorithms

## Cell 12: Bayesian Bandits - Prior and Posterior

- Type: Interactive
- Visualization:
  - Single arm with Beta distribution
  - Show prior Beta(alpha, beta) and posterior Beta(alpha+s, beta+f)
  - Animate posterior updating as data arrives
- Interactive widget:
  - Slider for prior parameters alpha, beta
  - Button "Pull Arm" (generates success/failure)
  - Display: Successes s, Failures f
  - Display: Posterior distribution curve
- Display:
  - Probability density plot showing prior and posterior
  - Shaded area showing uncertainty
- Comment box: "Start with prior belief Beta(alpha, beta). Update with data to
  get posterior Beta(alpha+s, beta+f)."
- Purpose: Introduce Bayesian inference for bandits

## Cell 13: Thompson Sampling Algorithm

- Type: Interactive
- Visualization:
  - K arms with Beta posteriors
  - At each step: sample theta_i from each posterior, pick argmax
  - Show sampled values as dots on posterior curves
- Interactive widget:
  - Slider for number of arms K (2-5)
  - Button "Run Thompson Sampling"
  - Speed slider
  - Display: Posterior distributions updating
- Display:
  - K probability density curves (posteriors)
  - Dots showing sampled theta_i at current step
  - Arrow pointing to selected arm
- Comment box: "Sample from each posterior, pick the arm with highest sample.
  This is Thompson Sampling."
- Purpose: Demonstrate Thompson Sampling's randomized decision-making

## Cell 14: Thompson Sampling - Probability Matching

- Type: Interactive
- Visualization:
  - K arms showing Pr(arm i is optimal | data)
  - Bar chart showing selection probabilities
  - Compare with empirical selection frequencies
- Interactive widget:
  - Button "Run 1000 Steps"
  - Display: Theoretical probability each arm is optimal
  - Display: Empirical frequency each arm is selected
- Display:
  - Two bar charts side-by-side
  - Left: Pr(arm i is optimal)
  - Right: Frequency arm i is selected
  - Show they match (probability matching property)
- Comment box: "Thompson Sampling selects each arm proportional to probability
  it's optimal."
- Purpose: Illustrate probability matching property of Thompson Sampling

## Cell 15: UCB vs Thompson Sampling Comparison

- Type: Interactive
- Visualization:
  - Split screen showing UCB and Thompson Sampling side-by-side
  - Same bandit instance for both
  - Show arm selections and regret curves
- Interactive widget:
  - Slider for number of arms K
  - Slider for suboptimality gaps Delta_i
  - Button "Run Both Algorithms"
  - Display: Regret curves for both
- Display:
  - Upper panels: UCB indices vs Thompson Sampling posteriors
  - Lower panels: Cumulative regret curves
  - Highlight similar performance
- Comment box: "Both achieve O(log T) regret. Thompson Sampling often has better
  constants in practice."
- Purpose: Compare the two best algorithms empirically

## Cell 16: Non-Stationary Bandits

- Type: Interactive
- Visualization:
  - 3 arms with time-varying means
  - Show mu_i(t) changing over time (e.g., sinusoidal or drift)
  - Algorithm must adapt to changing distributions
- Interactive widget:
  - Toggle "Stationary" vs "Non-Stationary"
  - Slider for change rate
  - Button "Run Algorithm" (UCB, Discounted UCB, Sliding Window)
  - Display: True means over time vs estimated means
- Display:
  - Line plots showing mu_i(t) over time
  - Overlay estimated means
  - Show when algorithm fails to adapt
- Comment box: "When means change over time, standard UCB fails. Need adaptive
  methods like discounted UCB."
- Purpose: Introduce non-stationary setting and need for adaptation

## Cell 17: Real-World Application - A/B Testing

- Type: Interactive
- Visualization:
  - Website with 3 design variants (A, B, C)
  - Each variant has unknown click-through rate
  - Show visitors arriving and clicking/not clicking
- Interactive widget:
  - Slider for true CTR of each variant
  - Button "Run A/B Test" (using Thompson Sampling)
  - Display: Visitors assigned to each variant
  - Display: Clicks observed, CTR estimates
- Display:
  - Bar chart showing CTR estimates with confidence intervals
  - Line plot showing cumulative clicks
  - Highlight winning variant
- Comment box: "Multi-armed bandits minimize opportunity cost during A/B testing
  by adaptively allocating traffic."
- Purpose: Connect theory to real-world application in web optimization

## Cell 18: Real-World Application - Clinical Trials

- Type: Interactive
- Visualization:
  - 3 treatment arms with unknown efficacy rates
  - Show patients arriving and receiving treatment
  - Success/failure outcomes visualized
- Interactive widget:
  - Slider for true efficacy of each treatment
  - Button "Run Adaptive Trial" (using Thompson Sampling)
  - Display: Patients assigned to each treatment
  - Display: Success rates
  - Ethical metric: Total successes vs fixed allocation
- Display:
  - Bar chart showing patient allocation over time
  - Line plot showing cumulative successes
  - Comparison with equal allocation baseline
- Comment box: "Adaptive trials using bandits maximize patient benefit while
  learning which treatment is best."
- Purpose: Show ethical importance of bandits in medical applications

## Cell 19: Contextual Bandits Preview

- Type: Interactive
- Visualization:
  - Arms have context-dependent rewards
  - Show context vector (e.g., user features) affecting arm choice
  - Example: Recommend movie based on user age, genre preference
- Interactive widget:
  - Slider for user age, genre preference
  - Button "Get Recommendation"
  - Display: Predicted reward for each movie given context
- Display:
  - Table showing context features
  - Bar chart showing predicted rewards
  - Arrow pointing to selected movie
- Comment box: "Contextual bandits use side information (context) to make better
  decisions. This is beyond basic MAB."
- Purpose: Preview extension to contextual bandits

## Cell 20: Summary and Key Takeaways

- Type: Markdown
- Content:
  - Key concepts:
    - Exploration-exploitation tradeoff is fundamental
    - Greedy fails, need to explore
    - Epsilon-greedy is simple but suboptimal (linear regret)
    - UCB achieves optimal O(log T) regret using optimism
    - Thompson Sampling achieves optimal regret using Bayesian inference
    - Regret measures opportunity cost of learning
  - When to use each algorithm:
    - Epsilon-greedy: Simple baseline, easy to implement
    - UCB: Frequentist setting, no tuning needed
    - Thompson Sampling: When prior knowledge available, often best empirical
      performance
  - Applications: Online advertising, A/B testing, clinical trials,
    recommendation systems
  - Extensions: Contextual bandits, non-stationary bandits, adversarial bandits
- Purpose: Consolidate learning and provide practical guidance
