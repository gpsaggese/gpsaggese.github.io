# `racket_strategy_PR4` Implementation Spec: Zero-Sum Placement Game

- `PR4` of `plan.racket_strategy.md`: implement paper Section V-A, the placement
  decision as a zero-sum game between striker and returner
- Scope: `racket_game.py` (payoff matrix, equilibrium solver, exploitability)
- Roadmap position: depends on `PR3`; `PR5` and `PR6` call it
- This is a specification only: no code in this document has been implemented

## Design Decisions

- **Payoff built from the scoring layer, with a continuation value**
  - Fact: Section V-A calls $u(c, c')$ a point-win probability (aim at $c$,
    returner recovered toward $c'$), but defines only $S$, an outright winner
  - Decision: $u(c, c') = \max_\ell P_{\mathrm{in}}(c,\ell)\,[1 - R(c,\ell \mid
    c') + R(c,\ell \mid c')\,p_{\mathrm{cont}}]$, with `continuation_win_prob`
    $p_{\mathrm{cont}} = 0$ by default, which gives back the paper's $S$
  - Why: the smallest bridge from $S$ to a point-win probability without a
    multi-shot model
- **Reuse the Monte Carlo step**: `PR3` `estimate_launch_table()` runs once;
  `compute_reachability()` is vectorized over the $N$ recovery positions, so the
  payoff costs $O(MLN)$ on top of one $O(MLK)$ sampling pass
- **Recovery positions are their own set**
  - Fact: the paper uses the target grid for both players ($\Delta(\mathcal{C})$)
  - Decision: `recovery_xy` holds $N$ positions, default the cell centers; the
    payoff is $M \times N$
  - Why: allows realistic returner sets (e.g., along the baseline) and the 3x3
    serve game of Walker and Wooders
- **Two explicit LPs with `scipy.optimize.linprog` (HiGHS)**
  - Striker: max $v$ s.t. $\sum_i \sigma_i u_{ij} \ge v\ \forall j$,
    $\sum_i \sigma_i = 1$, $\sigma \ge 0$; returner: the mirror min problem
  - Why: easier to review than dual marginals; cost is small next to sampling
  - Invariant: both LP values agree within `1e-6` (`hdbg.dassert`)
- **Exploitability of the pure argmax**: `compute_best_response_value()` gives the
  returner's best reply to any striker strategy, to test the Section V-A claim
  that always playing $c^\star$ is exploitable

## Trade-off and Alternative Design

- `nashpy`: extra dependency; support enumeration is exponential in $M$
- Fictitious play, multiplicative weights, double oracle: approximate or aimed
  at the repeated game of Section V-C; a fallback for very large $M \times N$
- Trade-off: an exact LP limits grid size, in exchange for exact equilibria

## Out of Scope

- Post-shot positioning game of Section V-C (future work iv)
- Minimax validation on real placement data (future work v)

## Current State

- Not applicable: no game code exists; `PR1` adds `scipy` to `requirements.txt`

## Implementation

### `racket_game.py`

- Interface (one new file):
  ```python
  class GameSolution:   # value: float, striker_strategy: pd.Series (cell_id),
      ...               # returner_strategy: pd.Series (recovery_id)
  def build_payoff_matrix(launch_table, returner, recovery_xy, *,
                          continuation_win_prob=0.0) -> pd.DataFrame: ...
  def solve_zero_sum_game(payoff: pd.DataFrame) -> GameSolution: ...
  def compute_best_response_value(payoff, striker_strategy) -> float: ...
  def make_pure_strategy(payoff, cell_id) -> pd.Series: ...
  ```

## Interaction with Existing Code

- Calls `racket_scoring.compute_reachability()`; no change to `PR3` code

## Configuration and Secrets

- Not applicable

## Unit Test Plan

- `test/test_racket_game.py`:
  - `Test_solve_zero_sum_game`:
    - Matching pennies `[[1, 0], [0, 1]]`: value 0.5, both strategies uniform
    - A strictly dominated row gets zero weight
    - A 1x1 payoff returns its single entry
    - Random 5x4 payoff (seed 0): every striker strategy in the support earns
      the game value against $\rho^\star$ (the Walker and Wooders prediction)
  - `Test_build_payoff_matrix`: 2 far-apart targets, recovery at each, zero
    error: off-diagonal 1, diagonal 0, and diagonal 0.5 with
    `continuation_win_prob=0.5`
  - `Test_compute_best_response_value`: pure argmax scores at most the game value

## Risks and Limitations to Call Out

- Equilibria may not be unique: tests check value and indifference only
- With a step-function $R$ the payoff is mostly 0 or $P_{\mathrm{in}}$, so
  supports can be sparse and grid-dependent
- 400 x 400 payoffs are fine for HiGHS; the reviewer checks runtime in `PR6`

## Result (to Fill in Once Implemented)

- TBD
