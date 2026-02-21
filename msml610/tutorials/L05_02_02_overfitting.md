# Bias-Variance Trade-Off: Interactive Visual Script

## Cell 1: True Target Function

- Purpose: Visualize the true target function that we want to learn
- Visualization:
  - Select the true function
    - slow sinusoid: f(x) = sin(0.5 * pi * x)
    - fast sinusoid: f(x) = sin(2* pi * x) for x in [-1, 1]
    - parabola: f(x) = 2*x^2 - 1 (scaled to [-1, 1] range)
    - constant: f(x) = 0
    - linear: f(x) = x
  - All functions bounded to [-1, 1] range
  - Add grid lines for better readability
  - Mark x and y axes clearly
- Interactive widget:
  - Random seed: controls noise generation
  - Select the true function
  - epsilon: std dev of noise (0 to 1)
- Display:
  - Plot the curve with optional noise visualization
  - Title: "True Target Function"
  - X-axis label: "x"
  - Y-axis label: "f(x)"
  - Y-axis limits: [-1.5, 1.5]
- Comment box: "This is the unknown target function we want to learn. In real-world problems, we don't have access to this complete curve - we only see a few sampled points."

## Cell 2: Constant Model (H_0)

- Purpose: Demonstrate learning with a constant hypothesis h(x) = b
- Visualization:
  - Same set-up as cell 1 (user should select the set up in cell 1 and then
    the same setup should hold across all the cells)
  - Use some global variables to keep the cells in sync
  - Plot the learned constant model as a horizontal line (green)
  - Shade the area between the constant and the true function to show
    approximation error
- Interactive widget:
  - Button to "Resample and Relearn" (new training points, new model  fit)
  - Display the learned parameter b
- Display:
  - One plot showing the in-sample data and the fitted model
  - One plot showing the out-sample data and the fitted model
  - One plot showing the true function and the fitted model
  - Show E_in and E_out value for current fit
  - Title: "Constant Model: h(x) = b"
- Comment box: "The constant model finds the best horizontal line to fit the 2
  points. It has high bias (poor approximation) but low variance (stable across
  different training sets)."

## Cell 5: Model Comparison - Multiple Realizations

- Purpose: Show how different training sets affect each model type
- Visualization:
  - Create 2 subplots side by side
  - Left subplot: Constant models from 20 different training sets (light green lines) with average (dark green)
  - Right subplot: Linear models from 20 different training sets (light orange lines) with average (dark orange)
  - Both show the true sinusoid (blue)
- Interactive widget:
  - Slider for number of realizations (5-50)
  - Button to "Regenerate All Models"
- Display:
  - Side-by-side comparison
  - Show variance value for each model type
  - Titles: "Constant Model - Multiple Realizations" and "Linear Model - Multiple Realizations"
- Comment box: "Notice how the constant models (left) cluster tightly together (low variance), while linear models (right) vary widely (high variance) depending on training points."

## Cell 6: Bias-Variance Decomposition Visualization

- Purpose: Visualize the components of error for both models
- Visualization:
  - Create bar chart with 2 bars (one for each model)
  - Each bar is stacked showing: bias^2 (bottom), variance (top)
  - Total height represents E_out
  - Use different colors for bias and variance components
- Interactive widget:
  - Display numerical values: bias^2, variance, and E_out for each model
- Display:
  - Stacked bar chart
  - Title: "Bias-Variance Decomposition: H_0 vs H_1"
  - X-axis: "Constant Model" and "Linear Model"
  - Y-axis: "Error"
  - Legend: "Bias^2" and "Variance"
- Comment box: "For learning (not just approximation), the constant model wins: E_out(H_0) = 0.5 + 0.25 = 0.75 < E_out(H_1) = 0.2 + 1.69 = 1.9. Lower bias doesn't guarantee better learning!"

## Cell 7: Different Target Functions - Line

- Purpose: Show bias-variance for a linear target function
- Visualization:
  - Plot target function: f(x) = 0.5*x (a straight line)
  - Plot 2 training points
  - Show constant model fit (horizontal line)
  - Show linear model fit (diagonal line)
- Interactive widget:
  - Slider for line slope (-2 to 2)
  - Slider for line intercept (-1 to 1)
  - Button to "Resample Training Points"
- Display:
  - All elements on one plot
  - Show E_out for both models
  - Title: "Linear Target Function"
- Comment box: "When the target is linear, the linear model has zero bias! It should perform much better than the constant model."

## Cell 8: Different Target Functions - Slow Sinusoid

- Purpose: Show bias-variance for a slow-varying sinusoid
- Visualization:
  - Plot target function: f(x) = sin(0.5 * pi * x)
  - Plot 2 training points
  - Show constant model fit
  - Show linear model fit
- Interactive widget:
  - Slider for sinusoid frequency (0.5 to 5)
  - Button to "Resample Training Points"
- Display:
  - All elements on one plot
  - Show E_out for both models
  - Title: "Slow Sinusoid Target Function"
- Comment box: "For a slow sinusoid, the linear model can approximate it reasonably well in the range [-1, 1]. The bias-variance tradeoff is less extreme."

## Cell 9: Different Target Functions - Parabola

- Purpose: Show bias-variance for a quadratic target function
- Visualization:
  - Plot target function: f(x) = x^2
  - Plot 2 training points
  - Show constant model fit
  - Show linear model fit
- Interactive widget:
  - Slider for parabola coefficient (-2 to 2)
  - Slider for parabola center (-1 to 1)
  - Button to "Resample Training Points"
- Display:
  - All elements on one plot
  - Show E_out for both models
  - Title: "Parabolic Target Function"
- Comment box: "The parabola is symmetric and smooth. Neither model can capture the curvature, so both have significant bias. The linear model still has higher variance."

## Cell 10: Different Target Functions - Fast Sinusoid

- Purpose: Show bias-variance for a high-frequency sinusoid
- Visualization:
  - Plot target function: f(x) = sin(5 * pi * x)
  - Plot 2 training points
  - Show constant model fit
  - Show linear model fit
- Interactive widget:
  - Slider for sinusoid frequency (1 to 10)
  - Button to "Resample Training Points"
- Display:
  - All elements on one plot
  - Show E_out for both models
  - Title: "Fast Sinusoid Target Function"
- Comment box: "With a fast-varying sinusoid, both models have very high bias. The complex target makes learning from just 2 points nearly impossible. This is deterministic noise!"

## Cell 11: Bias-Variance Curves - Model Complexity

- Purpose: Show how E_in and E_out change with model complexity
- Visualization:
  - Plot E_in curve (decreasing, can go to 0)
  - Plot E_out curve (U-shaped)
  - Plot bias curve (decreasing)
  - Plot variance curve (increasing)
  - Mark the optimal complexity point (minimum of E_out)
  - Shade "underfitting" region (left of minimum)
  - Shade "overfitting" region (right of minimum)
- Interactive widget:
  - Slider for number of training points N (2-100)
  - Dropdown for target function type (sinusoid, parabola, linear, fast sinusoid)
- Display:
  - All curves on same plot
  - X-axis: "Model Complexity (e.g., polynomial degree)"
  - Y-axis: "Error"
  - Vertical line marking optimal complexity
  - Title: "Bias-Variance Curves"
  - Legend for all curves
- Comment box: "As complexity increases: bias decreases, variance increases. E_out = bias^2 + variance has a sweet spot. More data (larger N) shifts the optimal point right, allowing more complex models."

## Cell 12: Regularization Path

- Purpose: Show how regularization parameter lambda affects model fit
- Visualization:
  - Plot true target function (sinusoid)
  - Plot training points
  - Plot learned model with current lambda value
  - Show the model getting simpler (more regularized) or more complex (less regularized)
- Interactive widget:
  - Slider for lambda on log scale (0.001 to 1000)
  - Display: "Small lambda = Complex Model = Low Bias, High Variance"
  - Display: "Large lambda = Simple Model = High Bias, Low Variance"
  - Show current values of bias^2, variance, E_out
- Display:
  - Main plot with all elements
  - Side panel showing error decomposition as stacked bars
  - Title: "Regularization: Finding Optimal lambda"
- Comment box: "Regularization parameter lambda controls model complexity. By varying lambda, we can explore the bias-variance tradeoff and find the optimal model."

## Cell 13: Impact of Training Set Size

- Purpose: Show how more data reduces overfitting
- Visualization:
  - Create 4 subplots for N = 2, 5, 10, 20 training points
  - Each subplot shows: true function, training points, learned model (high complexity)
  - Show E_in and E_out values for each N
- Interactive widget:
  - Slider for model complexity (polynomial degree 1-10)
  - Display the overfitting measure: (E_out - E_in) / E_out for each N
- Display:
  - 2x2 grid of subplots
  - Title: "More Data Reduces Overfitting"
  - Each subplot labeled with N value
- Comment box: "Rule of thumb: degrees of freedom = N / 10. With only 2 points, we should use at most 0.2 degrees of freedom (essentially a constant). With 20 points, we can safely use 2 degrees of freedom (linear model)."

## Cell 14: Noise and Overfitting

- Purpose: Show how noise level affects the bias-variance tradeoff
- Visualization:
  - Plot true noiseless function
  - Add stochastic noise: y = f(x) + epsilon
  - Show training points with noise
  - Show learned models at different complexity levels
- Interactive widget:
  - Slider for noise level sigma (0 to 1)
  - Slider for model complexity (polynomial degree 1-10)
  - Display E_out decomposition: variance + bias + noise
- Display:
  - Main plot showing noisy data and fit
  - Bar chart showing error decomposition
  - Title: "Stochastic Noise and Overfitting"
- Comment box: "Higher noise requires simpler models (more regularization). E_out = variance + bias^2 + noise. The noise term is irreducible - we can't learn it away."

## Cell 15: Deterministic vs Stochastic Noise

- Purpose: Show that complex target and noisy simple target are indistinguishable from training data
- Visualization:
  - Create 2 side-by-side subplots
  - Left: 10th order polynomial target + stochastic noise, 2nd order hypothesis
  - Right: 50th order polynomial target (noiseless), 2nd order hypothesis
  - Show N=15 training points on both
  - Show the learned 2nd order models
- Interactive widget:
  - Button to "Resample Training Points"
  - Slider for hypothesis complexity (2 to 10)
  - Display E_in and E_out for both scenarios
- Display:
  - Side-by-side comparison
  - Both plots show similar training data behavior
  - Title: "Can You Tell Them Apart?"
- Comment box: "From training data alone, we cannot distinguish between a noisy simple target and a noiseless complex target. Both act as 'noise' relative to our hypothesis set. This is deterministic noise!"

## Cell 16: Interactive Bias-Variance Exploration

- Purpose: Allow free exploration of bias-variance tradeoff with all parameters
- Visualization:
  - Main plot: target function, training points, learned model
  - Side plot: bias-variance decomposition bars
  - Bottom plot: bias-variance curves across complexity spectrum
- Interactive widget:
  - Dropdown for target function type (linear, slow sinusoid, parabola, fast sinusoid)
  - Slider for target complexity (frequency, polynomial degree, etc.)
  - Slider for number of training points N (2-100)
  - Slider for model complexity (1-10)
  - Slider for noise level sigma (0-1)
  - Button to "Resample Training Data"
  - Display all computed values: bias^2, variance, noise, E_in, E_out
- Display:
  - Three coordinated plots updating together
  - Title: "Interactive Bias-Variance Explorer"
- Comment box: "Experiment with different combinations! Key insights: (1) More data allows more complex models, (2) Match model complexity to data resources, not target complexity, (3) Noise requires simpler models."
