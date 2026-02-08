# Interactive Visual Script: Growth Function and VC Theory

## Cell 1: Dichotomy Explorer - 2D Perceptron with 3 Points

- Type: Interactive
- Purpose: Discover how points can be classified moving the hyperplane
- Visualization:
  - 2D plot with 3 points (A, B, C) that can be selected in different positions
    and different assignments
  - A slider/line representing a 2D perceptron (separating hyperplane) that can
    be rotated and moved
  - Points are colored inside based on which side of the line they fall (blue for
    +1, red for -1)
- Interactive widgets:
  - Select points in different positions
    - collinear1
    - collinear2
    - triangle1
    - triangle2
    - triangle3
  - Rotate slider: angle of separating line (0-360 degrees)
  - Position slider: offset of the line
- Display:
  - Current classification: "A: +1, B: -1, C: +1"

## Cell 2: Dichotomy Explorer - 2D Perceptron with 3 Points

- Type: Interactive
- Purpose: Discover that 3 points can be classified in 2^3 = 8 different ways by
  moving the line
- Visualization:
  - Same as Cell 1
  - 2D plot with 3 points (A, B, C) that can be selected in different positions
    and different assignments (represented by red or blue circles)
    - Points are colored inside based on which side of the line they fall (blue for
      +1, red for -1)
  - A slider/line representing a 2D perceptron (separating hyperplane) that can
    be rotated and moved
- Interactive widgets:
  - Select points in different positions and assignments
    - collinear
    - triangle
    - each with 8 different assignments
  - Rotate slider: angle of separating line (0-360 degrees)
  - Position slider: offset of the line
  - Button: to find the solution 
- Display:
  - Desired classification: "A: +1, B: -1, C: +1"
  - Current classification: "A: +1, B: -1, C: +1"
  - Print if the current dichotomy is achieved or not

## Cell 3: Dichotomy Explorer - 2D Perceptron with 4 Points

- Type: Interactive
- Purpose: Show that with 4 points, not all 16 classifications are possible -
  introduces the concept of break point
- Visualization:
  - Same as Cell 2 but with 4 points (A, B, C, D)
  - Start with points in a square configuration
- Interactive widgets:
  - Drag points A, B, C, D to different positions
  - Rotate slider: angle of separating line (0-360 degrees)
  - Position slider: offset of the line
  - Reset button to try different point configurations (circle, square, line,
    random)
- Display:
  - Current classification for all 4 points
  - Counter showing unique dichotomies found (should max out at 14)
  - Highlighted: "XOR pattern not achievable!" when points are in square and
    shows which 2 classifications are impossible
  - Desired dichotomy

<start>
## Cell 4: Dichotomy Explorer - Positive Rays

- Type: Interactive
- Purpose: Show simplest example where growth function is linear (N+1)
- Visualization:
  - Same set up as cell3_dichotomy_explorer_4points
  - 1D number line with N points placed on it
  - A vertical threshold line 'a' that can be moved
  - Points to the right of 'a' are colored blue (+1), left are red (-1)
- Interactive widgets:
  - Slider for N (number of points): 1 to 10
  - Slider to move threshold 'a' along the line
- Display:
  - Current dichotomy
  - Desired dichotomy

## Cell 5: Dichotomy Explorer - Positive Intervals

- Type: Interactive
- Purpose: Show example where growth function is quadratic
- Visualization:
  - Same set up as cell3_dichotomy_explorer_4points
  - 1D number line with N points
  - Two vertical threshold lines [a, b] that define an interval
  - Points inside [a, b] are blue (+1), outside are red (-1)
- Interactive widgets:
  - Slider for N (number of points): 1 to 8
  - Slider to move left boundary 'a'
  - Slider to move right boundary 'b'
- Display:
  - Current dichotomy
  - Desired dichotomy

## Cell 6: Dichotomy Explorer - Convex Sets

- Type: Interactive
- Purpose: Show example with exponential growth
- Visualization:
  - Same set up as cell3_dichotomy_explorer_4points
  - 2D plot with N points arranged in a circle
  - Ability to select any subset of points
  - Draw a convex hull around selected points (shaded region represents +1)
- Interactive widgets:
  - Slider for N (number of points): 3 to 8
  - "Random dichotomy" button - randomly selects points and shows convex hull
- Display:
  - Current dichotomy
  - Desired dichotomy
<end>

## Compute m_H(N) with brute force


## Cell 8: Growth Function Comparison

- Type: Interactive
- Visualization:
  - Line plot with N on x-axis (1 to 20) and m_H(N) on y-axis (log scale)
  - Multiple curves for different hypothesis sets:
    - Positive rays: m_H(N) = N + 1 (linear)
    - Positive intervals: m_H(N) ~ N^2 (quadratic)
    - 2D Perceptron: m_H(N) bounded by polynomial
    - Convex sets: m_H(N) = 2^N (exponential)
- Interactive widgets:
  - Toggle checkboxes to show/hide each hypothesis set
  - Slider for N range (1 to 50)
  - Highlight region where "learning is feasible" (polynomial growth)
- Display:
  - Legend showing each curve
  - Annotations:
    - "Polynomial growth -> Learning is possible"
    - "Exponential growth -> Overfitting risk"
  - Table of break points for each hypothesis set
- Purpose: Compare growth functions and understand why polynomial growth enables
  learning

## Cell 9: Shattering Demonstration

- Type: Interactive
- Visualization:
  - Grid of all 2^N possible labelings for N points
  - Visual indicator (checkmark or X) showing which labelings are achievable
  - Points displayed in a configurable arrangement
- Interactive widgets:
  - Slider for N (number of points): 1 to 4
  - Dropdown for hypothesis set (Positive rays, Positive intervals, 2D
    Perceptron, Convex sets)
  - Point arrangement selector (line, square, circle, random)
- Display:
  - Grid showing all 2^N classifications
  - Green checkmark for achievable dichotomies, red X for impossible ones
  - Count: "X out of 2^N dichotomies achievable"
  - "Shattering status: YES" or "NO (break point reached)"
- Purpose: Visual definition of shattering - can the hypothesis set achieve all
  possible labelings?

## Cell 10: Break Point Discovery

- Type: Interactive
- Visualization:
  - Bar chart showing m_H(N) vs 2^N for increasing N
  - Two bars per N value: actual m_H(N) in blue, 2^N in red
  - Highlight where they first diverge (break point)
- Interactive widgets:
  - Dropdown to select hypothesis set
  - Slider to step through values of N (1 to 10)
  - "Find break point" button that animates through N values
- Display:
  - Current N value
  - M_H(N) value and 2^N value
  - "Break point found at k = X" when m_H(k) < 2^k
  - Formula summary for selected hypothesis set
  - Explanation: "Break point exists -> Growth is polynomial -> Learning is
    feasible!"
- Purpose: Help students discover the break point through visual comparison

## Cell 11: 2D Perceptron - Complete Analysis

- Type: Interactive
- Visualization:
  - 2D plot with draggable points
  - All possible dichotomies displayed as a grid of small plots
  - Each small plot shows the points with one classification
  - Achievable dichotomies have a green border, impossible ones have red border
- Interactive widgets:
  - Slider for N: 1 to 5
  - Drag points to change configuration
  - "Optimal placement" button to arrange points to maximize dichotomies
  - "Random placement" button
- Display:
  - Current m_H(N) for the placement
  - Maximum possible m_H(N) for this N
  - Table: N | m_H(N) | 2^N | Shatters?
    - 1 | 2 | 2 | Yes
    - 2 | 4 | 4 | Yes
    - 3 | 8 | 8 | Yes
    - 4 | 14 | 16 | No
  - "Break point k = 4"
  - "VC dimension d_VC = 3"
- Purpose: Complete exploration of 2D perceptron growth function

## Cell 12: Why Break Points Matter for Learning

- Type: Interactive
- Visualization:
  - Two side-by-side plots:
    - Left: Hoeffding bound with M hypotheses: 2M exp(-2 epsilon^2 N)
    - Right: VC bound with growth function: 4 m_H(2N) exp(-1/8 epsilon^2 N)
  - X-axis: N (number of training examples)
  - Y-axis: Probability of bad generalization (log scale)
- Interactive widgets:
  - Slider for epsilon (error tolerance): 0.01 to 0.5
  - Slider for M (number of hypotheses): 10 to 10000
  - Dropdown for hypothesis set (affects m_H)
  - Slider for delta (probability threshold): 0.01 to 0.2
- Display:
  - Hoeffding bound curve (red, very loose)
  - VC bound curve (blue, much tighter)
  - Crosshair showing required N for desired delta
  - Text: "With break point -> polynomial growth -> bound eventually becomes
    useful"
  - Text: "Without break point -> exponential growth -> bound is useless"
  - Rule of thumb displayed: "N >= 10 \* d_VC for good generalization"
- Purpose: Connect break points to the feasibility of learning via
  generalization bounds

## Cell 13: N vs D Trade-Off Visualization

- Type: Interactive
- Visualization:
  - Contour plot or heatmap
  - X-axis: d_VC (VC dimension, 1 to 50)
  - Y-axis: N (number of training examples, 10 to 1000)
  - Color intensity: Probability of good generalization
  - Green region: "Safe zone" where N >= 10 \* d_VC
  - Yellow region: "Risky zone" where N < 10 \* d_VC
  - Red region: "Danger zone" where N << d_VC
- Interactive widgets:
  - Slider for epsilon (error tolerance)
  - Slider for delta (confidence level)
  - Click on plot to see specific (d_VC, N) pair analysis
- Display:
  - Selected point: (d_VC = X, N = Y)
  - Generalization bound: E_out <= E_in + Omega(N, H, delta)
  - Omega value calculated
  - Recommendation: "Increase N by Z examples" or "Reduce model complexity by W"
- Purpose: Practical guidance on choosing model complexity given dataset size

## Cell 14: Interactive Model Selection Simulator

- Type: Interactive
- Visualization:
  - Top: True underlying function (unknown to learner) plotted
  - Middle: Training data points sampled from true function with noise
  - Bottom: Three hypothesis sets with different d_VC trying to fit the data
    - Simple model (d_VC = 2): underfitting
    - Right-sized model (d_VC ~ 5): good fit
    - Complex model (d_VC = 20): overfitting
- Interactive widgets:
  - Slider for N (training points): 10 to 200
  - Slider for noise level: 0 to 0.5
  - Button "Generate new dataset"
  - Toggle "Show test set" to reveal test error
- Display:
  - For each model:
    - E_in (training error)
    - E_out (test error) when toggled
    - VC bound: E_out <= E_in + Omega
    - Color code: Green if bound is satisfied, Red if violated
  - Highlight which model achieves best generalization
  - Explanation: "Model 2 balances complexity and generalization"
- Purpose: Tie together all concepts - show why growth function and break points
  matter for practical learning
