### [ ] Implement the Optimal Shot Placement Framework for Racket Sports

* Repo: <Which repos are affected>
- [ ] helpers (https://github.com/causify-ai/helpers)
- [x] umd_classes (https://github.com/gpsaggese/gpsaggese.github.io)

* Problem
- `papers/Optimal_strategy_for_racket_sports/paper.md` proposes a reduced-order
  framework for shot placement in tennis and pickleball, but no code implements it
  - Section IX lists the implementation as future work (i)
  - Section VI is hand-computed with illustrative $P_{\mathrm{in}}$ values
  - `papers/Optimal_strategy_for_racket_sports/figures/make_figures.py` hard-codes
    the physics and the toy scores inline
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

- [ ] PR2: Implement the closed-form trajectory model of Section III-C (see
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

- [ ] PR6: Add the example notebook with computational results for Sections VI-VIII
      (see spec.racket_strategy_PR6.md)
  - Depends on: PR5
  - Add `racket_strategy.example.ipynb` / `.py` and the experiment helpers
  - Export result figures and tables to `results/`

- [ ] PR7: Report the computational results in the paper
  - Depends on: PR6
  - No spec: prose update whose content is fixed by the PR6 outputs
  - Add a "Computational Results" section to `paper.md` with the PR6 figures
  - Fix the $P_{\mathrm{in}}$ formula of Section IV-B and describe the aim-angle
    choice
  - Update the abstract, the contributions, Section VI caveat, and Section IX
  - Copy the figures to `papers/Optimal_strategy_for_racket_sports/figures/` and
    update the `Makefile` dependencies
