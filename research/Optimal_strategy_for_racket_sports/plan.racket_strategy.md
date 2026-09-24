### [ ] Implement the Optimal Shot Placement Framework for Racket Sports

* Repo: <Which repos are affected>
- [ ] helpers (https://github.com/causify-ai/helpers)
- [x] umd_classes (https://github.com/gpsaggese/gpsaggese.github.io)

* Problem
- `papers/Optimal_strategy_for_racket_sports/paper.md` proposes a reduced-order
  framework for shot placement in tennis and pickleball
- Goal: a tested package in `research/Optimal_strategy_for_racket_sports/` that:
  - Implements Sections III-VII of the paper
  - Produces computational results that replace the hand-worked example
- Out of scope (paper future work, follow-up issues):
  - Calibration of $\sigma_\theta, \sigma_v, \sigma_\phi, t_r, v_p$ from tracking
    data (ii)
  - Graded, probabilistic reachability (iii)
  - Post-shot positioning as a repeated game (iv)
  - Doubles geometry and minimax validation on rally data (v)
  - Drag and spin bias quantification (Section VIII)

* Model
- 2D court-fixed frame, already implemented in `racket_params.py`: origin at
  the net center on the ground, `x` lateral, `y` along the court (striker side
  `y < 0`, returner side `y > 0`)
- Striker: at a fixed position `(x_s, y_s)` on their side of the court, hits
  the ball
- Returner: at a fixed starting position `(x_r, y_r)` on the other side
  - For the first shot of a point, the starting position is the serve
    position (`get_service_box_region()`)
  - For later shots in a rally, the starting position is wherever the
    previous shot left the returner (out of scope until `PR7`'s example
    notebook)
- Shot: the striker aims at a target point `(x, y)` on the returner's side;
  execution error (`ShotErrorModel`) gives the actual landing a Gaussian
  scatter around `(x, y)`, already implemented in
  `racket_trajectory.get_feasible_launches()` / `simulate_landings()` (`PR2`)
- `PR3`'s grid scoring builds directly on this: each grid cell is a candidate
  target `(x, y)`; $P_{\mathrm{in}}$ is the in-bounds fraction of the landing
  scatter around it, and reachability $R$ compares the returner's travel time
  from `(x_r, y_r)` to the cell against the ball's flight time (a function of
  the ball's speed)
- The composite score $S(c) = P_{\mathrm{in}}(c)\,(1 - R(c))$ is exactly the
  "probability B misses the shot" (a winner) the user cares about; `PR6`'s
  notebook exposes this as a heatmap over every candidate `(x, y)`, driven by
  click-to-run controls for ball speed, player move speed, shot std dev, and
  both players' positions

* Info
- **Type**: feature
- **Reason of the problem**: the paper is a methodology only: no implementation or
  evaluation exists
- **Confidence in the fix**: medium
  - High for the physics (closed form) and the scoring (paper formulas)
  - Medium for 3 choices the paper leaves open, decided in the specs:
    - $P_{\mathrm{in}}$ counts in-bounds landings, not landings in cell $c$
      (Section IV-B formula and text disagree): `spec.racket_strategy_PR3.md`
    - The aim angle is the best feasible angle per cell (paper does not pick one
      point of $\mathcal{F}(T)$): `spec.racket_strategy_PR2.md`
    - Game payoff $u$ adds a continuation value to $S$ (paper calls $u$ a
      point-win probability but only defines $S$): `spec.racket_strategy_PR4.md`
- **Fix complexity**: medium
- **Verification plan**:
  - Unit tests reproduce the paper numbers:
    - Figure 1: $v_0 = 22.9$ m/s, $T_f = 0.88$ s, net clearance 0.40 m
    - Table II: $R$ and $S$ for both sports, argmax $c_3$ (tennis), $c_1$
      (pickleball)
  - `pytest research/Optimal_strategy_for_racket_sports/test` passes
  - Both notebooks run end to end in Docker
  - `make` in `papers/Optimal_strategy_for_racket_sports/` rebuilds `paper.pdf`

* Solution

- Module layering (each layer imports only the layers above it):
  - `racket_params.py`: sport, player, error parameters and court geometry
  - `racket_trajectory.py`: closed-form 1D/2D ball flight and error propagation
  - `racket_scoring.py`: grid, Monte Carlo $P_{\mathrm{in}}$, reachability, score
  - `racket_game.py`: zero-sum placement game
  - `racket_strategy_utils.py`: plots and experiment helpers for the notebooks

- [x] PR1: Scaffold the project and encode the sport parameters (see
      spec.racket_strategy_PR1.md)
  - Depends on: none
  - Copy the Docker files from `class_project/project_template/`
  - Add `racket_params.py` with Table I presets `TENNIS`, `PICKLEBALL`
  - Add `README.md` with quick start and the module table above
  - Add `test/test_racket_params.py`

- [x] PR2: Implement the closed-form trajectory model of Section III-C (see
      spec.racket_strategy_PR2.md)
  - Depends on: PR1
  - Add `racket_trajectory.py`: launch speed solver, feasible set, error sampling,
    vectorized landing simulation
  - Add `test/test_racket_trajectory.py` reproducing the Figure 1 numbers

- [ ] PR3: Implement grid-based Monte Carlo scoring for rally and serve, Section IV
      (see spec.racket_strategy_PR3.md)
  - Depends on: PR2
  - Add `racket_scoring.py`: target grid, opponent-independent launch table,
    reachability, composite score, argmax, serve situation
  - Add `test/test_racket_scoring.py` reproducing Table II

- [ ] PR4: Implement the zero-sum placement game of Section V-A (see
      spec.racket_strategy_PR4.md)
  - Depends on: PR3
  - Add `racket_game.py`: payoff matrix, LP solver, best-response value
  - Add `test/test_racket_game.py`

- [ ] PR5: Add the plotting utils and the API notebook (see
      spec.racket_strategy_PR5.md)
  - Depends on: PR2, PR3, PR4
  - Add `racket_strategy_utils.py` and `racket_strategy.API.ipynb` / `.py`
  - Add `test/test_racket_strategy_utils.py` and
    `test/test_docker_racket_strategy.py`

- [ ] PR6: Add an interactive shot-placement exploration notebook (see
      spec.racket_strategy_PR6.md)
  - Depends on: PR3, PR5
  - Add `racket_strategy.exploration.ipynb` / `.py`: click-to-run controls for
    ball speed, player move speed, shot std dev (`error_scale`), and both
    players' positions, driving `P_in`, `1 - R`, and `S` heatmaps
  - Add `build_exploration_widget()` to `racket_strategy_utils.py`
  - Extend `test/test_racket_strategy_utils.py` and
    `test/test_docker_racket_strategy.py`

- [ ] PR7: Add the example notebook with computational results for Sections VI-VIII
      (see spec.racket_strategy_PR7.md)
  - Depends on: PR5
  - Add `racket_strategy.example.ipynb` / `.py` and the experiment helpers
  - Export result figures and tables to `results/`

- [ ] PR8: Report the computational results in the paper
  - Depends on: PR7
  - No spec: prose update whose content is fixed by the PR7 outputs
  - Add a "Computational Results" section to `paper.md` with the PR7 figures
  - Fix the $P_{\mathrm{in}}$ formula of Section IV-B and describe the aim-angle
    choice
  - Update the abstract, the contributions, Section VI caveat, and Section IX
  - Copy the figures to `papers/Optimal_strategy_for_racket_sports/figures/` and
    update the `Makefile` dependencies
