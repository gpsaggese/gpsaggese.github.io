# Implement notebook script

notebook = `msml610/tutorials/L05_02_01_bias_variance.ipynb`
script = `msml610/tutorials/L05_02_01_bias_variance.md`

## Build notebook
- If the $notebook doesn't exist 
  ```bash
  > cp ./msml610/tutorials/template.ipynb $notebook
  > git add $notebook
  ```
- Execute `docs/ai_prompts/notebooks.implement_script.md` for cells between
  <start> and <end> of $script
- Update $notebook

# Bias-Variance: Interactive Visual Script

## Cell 1: Approximation

- Purpose: Show which model between a constant and a linear model approximate the
  known function
    f(x) = sin(pi * x) for x in [-1, 1]
- Visualization:
  - Plot the target function
  - All functions bounded to [-1, 1] range
  - Add grid lines for better readability
  - Mark x and y axes clearly
- Fit a constant model g_0 and a line g_1 to the sinusoid
- Display:
  - Plot 3 subgraphs
    - one with the true function and the constant model (and the approximation
      error)
    - one with the true function and the linear model (and the approximation
      error)
    - one with a comment
  - Title: "Approximation"
  - X-axis label: "x"
  - Y-axis label: "f(x)"
  - Y-axis limits: [-1.5, 1.5]
- Comment box: 
  - Print the error for both models
  - Show that the linear model fits better

## Cell 2: Learning Once

- Purpose: Show learning and approximation are not the same thing
- Visualization:
    f(x) = sin(pi * x) for x in [-1, 1]
  - Same set-up of cell 1
- This time pick two random points from the sinusoid
  - Add a seed widget using the function in msml610, which controls
    which points are generated
  - Add a N_samples points to control how many points are generated
    (default = 2) in the training set
  - Fit a constant model g_0 and a line g_1 to the training set
- Display:
  - Plot 3 subgraphs
    - one with the true function and the fit constant model (with E_in and the
      out of sample error E_out)
    - one with the true function and the fit linear model (with E_in and the OOS
      error E_out)
    - one with a comment
  - Title: "Learning from N points"
  - X-axis label: "x"
  - Y-axis label: "f(x)"
  - Y-axis limits: [-1.5, 1.5]
- Comment box: 
  - Print the in-sample error E_in for both models
  - Print the OOS error E_out for both models

## Cell 3: Learning over multiple training set

- Purpose: Show bias and variance decomposition
- Visualization:
  - Same set-up of cell 2
- This time pick two random points from the sinusoid
  - Create a widget with N_samples = 2
  - Create a widget with N_experiments = 20
  - Fit a constant model g_0 and a line g_1 to the in sample with N_samples
    points and show all the fitted models over N_experiments, using alpha=.5
- Display:
  - Plot 3 subgraphs
    - one with the true function and the constant models
    - one with the true function and the linear models
    - one with a comment
  - Title: "Learning"
  - X-axis label: "x"
  - Y-axis label: "f(x)"
  - Y-axis limits: [-1.5, 1.5]
- Comment box: 
  - Print the in-sample error for both models averaged over the N_experiments
  - Print the OOS error for both models over the N_experiments

## Cell 4: Learning plots (Bias-Variance)

- Purpose: Compute bias and variance decomposition changing various variables
- Visualization:
  - Same set-up of cell 3
- Display:
  - seed is fixed
  - Plot 3 subgraphs
    - one with the true function and the constant models
    - one with the true function and the linear models
    - one with a comment
  - Show E_in, E_out, bias, variance for the constant and linear model
    as function 
  - Title: "Learning"
  - X-axis label: "x"
  - Y-axis label: "f(x)"
  - Y-axis limits: [-1.5, 1.5]
- Comment box: 
  - Print the in-sample error for both models averaged over the N_experiments
  - Print the OOS error for both models over the N_experiments

<start>
## Cell 5: Learning plots with noise

- Purpose: Same as Cell 3 but also add a widget to control the amount of Gaussian
  noise
- Reuse the same code as cell3_learning_bias_variance but make sure that the
  previous code is not changed. In practice for Cell 3 there is no noise widget
  and the amount of noise is set to 0

## Cell 6: Learning plots with noise (Bias-Variance)

- Purpose: Same as Cell 4 but also add a widget to control the amount of Gaussian
  noise
- Reuse the same code as cell4_learning_bias_variance but make sure that the
  previous code is not changed. In practice for Cell 4 there is no noise widget
  and the amount of noise is set to 0
<end>
