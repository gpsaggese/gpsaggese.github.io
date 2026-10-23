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

* Vision
- GP's ideas for applying this research
  ([#664](https://github.com/gpsaggese/gpsaggese.github.io/issues/664)):
  - An academic paper (`papers/Optimal_strategy_for_racket_sports/paper.md`)
  - Mainstream press coverage, e.g., "UMD research solves tennis"
  - A puzzle-style training app: players see a scenario ("player 1 here, player 2
    there") and choose the optimal shot, to train their intuition, like chess
    puzzles

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

- [x] PR3: Implement grid-based Monte Carlo scoring for rally and serve, Section IV
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
  - Partially done ahead of `PR4`: `draw_court()`, `plot_trajectory_fan()`,
    `plot_court_heatmap()`, `plot_launch_tradeoff()`, `build_score_widget()`,
    and the API notebook's Parts 1-7 (sport parameters through serve) are
    implemented, to unblock `PR6`; `plot_mixed_strategy()` and the
    notebook's Part 8 (placement game) still need `PR4`'s `racket_game.py`

- [x] PR6: Add an interactive shot-placement exploration notebook (see
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

* Milestones
- The milestones of #590, defined by GP in the
  [checklist comment](https://github.com/gpsaggese/gpsaggese.github.io/issues/590#issuecomment-6075690143)
  - A simpler 2D track: the uncertainty sits directly on the landing point, without
    the trajectory model of `PR2` and `PR3`

- [x] Milestone 1: probability that a return lands in
  - Done in [#619](https://github.com/gpsaggese/gpsaggese.github.io/pull/619), with
    the follow-up [#654](https://github.com/gpsaggese/gpsaggese.github.io/pull/654)
  - Player 1 serves from the deuce or ad side to (`x1`, `y1`), following the serve
    rules; player 2 moves there and returns toward (`x2`, `y2`) with a 2D Gaussian
    spread (`std_x2`, `std_y2`)
  - $P_{\mathrm{in}}$ in closed form, checked by Monte Carlo, in
    `notebooks/racket_shot_probability_utils.py` and
    `notebooks/racket_shot_probability.ipynb`

- [-] Milestone 2: probability of a winning return
      ([#624](https://github.com/gpsaggese/gpsaggese.github.io/issues/624)), as
      agreed with GP in
      [#690](https://github.com/gpsaggese/gpsaggese.github.io/issues/690); in
      progress in [PR #715](https://github.com/gpsaggese/gpsaggese.github.io/pull/715)
  - Sources:
    [GP's answers on #624](https://github.com/gpsaggese/gpsaggese.github.io/issues/624#issuecomment-6186123404),
    [GP's answers on #690](https://github.com/gpsaggese/gpsaggese.github.io/issues/690#issuecomment-6252896405),
    and
    [the agreed plan](https://github.com/gpsaggese/gpsaggese.github.io/issues/690#issuecomment-6260766491)
  - One-shot setting: player 1 serves, player 2 returns
  - Player 1 serves with the Milestone 1 rules (a fault means no return), then
    stays at the center of the baseline, `(0, -L/2)`
  - Player 2 stands at the serve landing point (`x1`, `y1`) and aims at
    (`x2`, `y2`); the return lands with the same 2D Gaussian as Milestone 1
  - Flight time: for each sampled landing point $(X, Y)$, $T_f$ is the distance
    from (`x1`, `y1`) to $(X, Y)$ divided by the ball speed (a widget)
    - Why: the aim point is a latent variable, while the landing point is where
      the ball actually goes (GP on #690)
  - Player 1 reacts for a constant time $t_r$, the time to decide where the ball
    will land, then moves at a constant speed $v_p$, and reaches the ball if the
    distance from `(0, -L/2)` to $(X, Y)$ is at most $v_p \max(0, T_f - t_r)$
  - Winner: the return lands in AND player 1 does not reach it, computed by
    Monte Carlo as one event on the same samples, with no product of
    probabilities (GP on #690)
  - Defaults: ball speed 25 m/s for tennis and 12 m/s for pickleball,
    $t_r = 0.2$ s, $v_p = 1.5$ m/s, each with a slider
  - A heatmap of the winner probability over every target (`x2`, `y2`) on player
    1's half, with the best target marked, for tennis and pickleball
  - Code: new functions in `notebooks/racket_shot_probability_utils.py` and a
    new Part 2 of `notebooks/racket_shot_probability.ipynb`, in the draft
    [PR #715](https://github.com/gpsaggese/gpsaggese.github.io/pull/715)
  - Decisions made while building it:
    - **Common random numbers**: every target of the heatmap gets
      `np.random.default_rng(seed)` with the same seed, so all targets use the
      same standard normal draws, shifted to their own aim point
      - Two targets differ because of where they aim, not because of sampling
        noise, so comparing them to find the best one is fair
    - **Best region and star**: many targets can tie at the top (e.g., every
      sample a winner, $P(\text{winner}) = 1$), so the best is a region
      - The region holds every target within 2 standard errors of the highest
        $P(\text{winner})$, and is outlined on the map
      - The star is the cell of the region closest to its centroid
    - **Grid and samples**: 10 x 15 cells for tennis and 9 x 10 for pickleball,
      with 5000 samples per target, chosen so one map takes under 2 s
    - **Settings string**: the comments panel and a copyable line under the
      figure list every setting, e.g., `sport=Tennis side=deuce x1=-2.1 y1=3.2
      std_x2=0.5 std_y2=1.0 ball_speed=25.0 t_r=0.2 v_p=1.5 seed=1 n_x=10 n_y=15
      n_samples=5000`, so a result in the Gdoc can be reproduced exactly
      ([GP on #694](https://github.com/gpsaggese/gpsaggese.github.io/issues/694#issuecomment-6336301932))
    - **Bounce ignored**: player 1 must reach the landing point by the time the
      ball lands; the bounce is a 3D effect, and the 2D approximation is good
      enough for the first and second order effects
      ([GP on #690](https://github.com/gpsaggese/gpsaggese.github.io/issues/690#issuecomment-6336175829))
    - **Spread independent of ball speed**: kept as a first approximation, so in
      this model hitting as fast as possible always looks best; GP filed
      [#723](https://github.com/gpsaggese/gpsaggese.github.io/issues/723)
  - Finding: with the defaults, the best return is the middle of player 1's half,
    not a spot near the lines
    - Player 1 covers only about 0.6 m in tennis: a return landing next to them
      flies about 15 m in 0.61 s at 25 m/s, and $1.5 \cdot (0.61 - 0.2) \approx
      0.6$ m
    - So $P(\text{winner})$ is almost $P_{\mathrm{in}}$, except in a small dip
      around player 1, and the middle keeps the most returns in

* Open Points
- From
  [GP's answers on #624](https://github.com/gpsaggese/gpsaggese.github.io/issues/624#issuecomment-6186123404):
  - **Speed vs accuracy**: the spread does not depend on the ball speed yet, so
    hitting as fast as possible always looks best, since it leaves player 1 the
    least time
    - There is no clear model: ideally it is estimated by experiments, and it
      differs with the player's level
    - GP filed [#723](https://github.com/gpsaggese/gpsaggese.github.io/issues/723)
      for it; Milestone 2 keeps the spread independent of the ball speed
  - **Optimal positioning**: where each player should stand, as a function of the
    ball and of the other player, is left to the RL algorithm later
    - GP was taught to gravitate toward the center, but does not believe it is
      optimal
    - In closed form, the optimal position maximizes the probability of reaching
      the ball given the opponent's hitting position: it can be solved numerically
      and compared with what RL finds by brute force
  - **Factorization**: whether the winner probability factorizes as
    $P_{\mathrm{in}} \cdot P(\text{not reached})$; GP believes some factorization
    may hold but is not sure
    - Resolved in
      [#690](https://github.com/gpsaggese/gpsaggese.github.io/issues/690#issuecomment-6252896405):
      Milestone 2 computes the winner probability as one event, with no product
- From
  [#690](https://github.com/gpsaggese/gpsaggese.github.io/issues/690#issuecomment-6322544771):
  - **Reaching after the bounce**: player 1 must reach the landing point by the
    time the ball lands; in real play they can also take the ball after the
    bounce, which gives them more time
    - Resolved in
      [#690](https://github.com/gpsaggese/gpsaggese.github.io/issues/690#issuecomment-6336175829):
      ignore the bounce, since it is a 3D effect
    - The reach rule lives only in `compute_is_reached()`, so a change stays
      there

* Long-Term Goal
- From
  [GP's answers on #624](https://github.com/gpsaggese/gpsaggese.github.io/issues/624#issuecomment-6186123404):
  - First, the best strategy when there is only a serve and a response: maximize
    winners (Milestone 2)
  - Then, maximize the probability of winning a game, with brute-force RL or an
    iterative solution built from the same blocks, e.g., optimizing the
    probability of winning one step ahead instead of discounting the whole
    trajectory

* Tasks
- GP's rule
  ([#625](https://github.com/gpsaggese/gpsaggese.github.io/issues/625#issuecomment-6185837906)):
  we only execute tasks that are on this master plan, after creating them in GH
- Status: `[x]` done, `[-]` in progress (open PR), `[ ]` not started

- [x] Milestone 1 of
      [#590](https://github.com/gpsaggese/gpsaggese.github.io/issues/590):
      [PR #619](https://github.com/gpsaggese/gpsaggese.github.io/pull/619)
- [x] [#623](https://github.com/gpsaggese/gpsaggese.github.io/issues/623):
      Milestone 1 follow-ups (renames, diagonal serve, "Fault" label),
      [PR #654](https://github.com/gpsaggese/gpsaggese.github.io/pull/654)
  - Manual widget checks of the Milestone 1 notebook are still pending
- [-] [#624](https://github.com/gpsaggese/gpsaggese.github.io/issues/624):
      Milestone 2, winner probability heatmap,
      [PR #715](https://github.com/gpsaggese/gpsaggese.github.io/pull/715)
- [-] [#625](https://github.com/gpsaggese/gpsaggese.github.io/issues/625):
      project log `log.racket_strategy.md`,
      [PR #660](https://github.com/gpsaggese/gpsaggese.github.io/pull/660)
- [ ] [#626](https://github.com/gpsaggese/gpsaggese.github.io/issues/626):
      doc on creating a PR with the Claude Code skills
- [-] [#627](https://github.com/gpsaggese/gpsaggese.github.io/issues/627):
      paper, add the author and the Milestone 1 model,
      [PR #661](https://github.com/gpsaggese/gpsaggese.github.io/pull/661)
- [ ] [#663](https://github.com/gpsaggese/gpsaggese.github.io/issues/663):
      research other papers about optimal tennis strategy
- [ ] [#664](https://github.com/gpsaggese/gpsaggese.github.io/issues/664):
      brainstorm how to apply this research (see Vision)
- [-] [#666](https://github.com/gpsaggese/gpsaggese.github.io/issues/666):
      update this master plan with the Milestone 2 discussion
- [ ] [#669](https://github.com/gpsaggese/gpsaggese.github.io/issues/669):
      meta-doc on how we run the research project
- [ ] [#670](https://github.com/gpsaggese/gpsaggese.github.io/issues/670):
      generate the paper figures from the notebooks
