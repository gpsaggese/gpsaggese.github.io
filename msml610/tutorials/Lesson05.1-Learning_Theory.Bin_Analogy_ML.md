# Understanding the Bin Analogy

### Cell 1: Visual Bin
- Type: Code + Interactive Output
- Visualization: Draw a 2D bin filled with red and green marbles
- Interactive widget: Slider for mu (true proportion of red marbles, 0-1)
- Display: Show bin with marbles colored proportionally to mu
- Purpose: Give students concrete visual of the "unknown" population
- Label: "Population: Unknown mu = [value]"

### Cell 2: Single Experiment - Is nu Close to mu?
Type: Code + Interactive
- Setup: Run single sampling experiment
- Interactive widgets:
  - mu slider: 0-1
  - N slider: 10-1000
  - "Run experiment" button
- Outputs:
  - Bar chart: Show mu (population) vs nu (sample) side-by-side
  - Display numerical values: |nu - mu|
  - Color code: Green if close (< 0.1), Yellow if medium (< 0.2), Red if far
- Insight box: "In this run, nu [was/wasn't] close to mu"

### Cell 3: Monte Carlo Simulation - Distribution of nu
Type: Code + Interactive Visualization
- Interactive widgets:
  - mu slider: 0-1
  - N slider: 10-500
  - n_experiments slider: 100-10000
  - "Run simulation" button
- Process: Run n_experiments sampling experiments, collect all nu values
- Visualizations:
  - Histogram of nu values (with KDE overlay)
  - Vertical line at mu (true value)
  - Shade regions: |nu - mu| > eps for user-selected eps
  - Show mean and std of empirical distribution
- Displays:
  - Empirical P(|nu - mu| > eps) = (fraction in shaded region)
  - Note concentration around mu
- Purpose: Build intuition that nu clusters around mu

### Cell 5: Using Hoeffding's Inequality
Type: Markdown
- Mathematical formulation:
  - P(|nu - mu| > eps) ≤ 2e^(-2eps²N)

- Interactive widgets:
  - mu slider: 0-1
  - N slider: 10-500
  - eps slider: 0.01-0.5
  - n_simulations slider: 1000-100000
- Process:
  - Run Monte Carlo to get empirical P(|nu - mu| > eps)
  - Calculate Hoeffding bound: 2e^(-2eps²N)
- Visualization:
  - Bar chart: Empirical probability vs Hoeffding bound
  - Always show: Empirical ≤ Bound (Hoeffding is conservative)
  - Color bars differently
- Comment box: "The bound guarantees the empirical never exceeds it"

### Cell 6: Connection to Machine Learning - The Bridge
Type: Markdown + Visualization
- The analogy breakdown:
  - Bin → Input space X
  - Marble → A point x ∈ X
  - Red marble → h(x) = f(x) (hypothesis correct)
  - Green marble → h(x) ≠ f(x) (hypothesis wrong)
  - mu → E_out(h) (out-of-sample error)
  - nu → E_in(h) (in-sample/training error)
  - Sample → Training dataset
- Visual diagram: Side-by-side comparison showing correspondence
- Key insight: "Hoeffding tells us training error tracks test error!"

### Cell 7: ML Validation - Single Hypothesis
Type: Code + Interactive
- select a random seed
- Setup: Generate synthetic binary classification data with 2 features
  - The decision boundary is a line with some noise
  - True function f (shown as decision boundary)
  - Hypothesis h (shown as decision boundary)
  - Points colored by f(x)
- Split data in in-sample and out-sample which is selected 60-40
- Interactive widgets:
  - N_train slider (training set size)
  - sigma is amount of noise
  - Adjust hypothesis h by two parameters shifting and rotating (this represents
    the training)
- Display:
  - Training set plot with decision boundaries
  - Calculate E_in (training error)
  - Hoeffding bound: P(|E_in - E_out| > eps) ≤ [value]
  - Confidence statement: "With 95% probability, E_out ∈ [E_in - eps, E_in + eps]"
- Purpose: Show how Hoeffding validates a single chosen model

### Cell 8: ML Learning - Multiple Hypotheses Problem
Type: Markdown
- The challenge: In learning, we choose h from hypothesis set H
- Why this matters:
  - Validation: test 1 coin (1 hypothesis) - Hoeffding applies
  - Learning: choose best of M coins (M hypotheses) - need union bound
- Mathematical setup:
  - P(|E_in(g) - E_out(g)| > eps for chosen g) ≤ 2M e^(-2eps²N)
  - Price of selection: factor of M
- The problem: M often infinite (e.g., all linear separators)

### Cell 9: Demo - Multiple Hypothesis Selection
Type: Code + Interactive
- Setup: Generate data, try M different hypotheses
- Interactive widgets:
  - N slider (dataset size)
  - M slider (number of hypotheses to try)
  - eps threshold
- Process:
  - Generate M random hypotheses
  - For each: calculate E_in and E_out (we know true f)
  - Choose h_best with minimum E_in
  - Check if |E_in(h_best) - E_out(h_best)| > eps
- Visualization:
  - Scatter: E_in vs E_out for all M hypotheses
  - Highlight chosen h_best
  - Show eps bands
  - Plot multiple runs to estimate empirical failure rate
- Compare:
  - Empirical P(bad generalization)
  - Union bound: 2M e^(-2eps²N)
  - Show union bound is conservative
- Purpose: Demonstrate the multiple hypothesis penalty
