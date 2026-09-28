# RL for Tennis / Pickleball Rally Strategy

## Status
**Status:**: in_progress
**Complete Specs:**: 50%
**Assignee:**: TBD

## Core Idea

- Train an RL agent to learn optimal rally strategy (shot selection, court
  positioning, adaptive tactics) purely from self-play, without hand-coding
  coaching heuristics
- Validate the result by checking whether the emergent policy resembles real
  coaching principles (e.g., exploit the opponent's weak side, be patient with
  the inferior player), which would show the reward and environment design are
  rich enough to induce genuine strategic behavior rather than degenerate
  policies
- Rather than model full 2D tennis or pickleball directly, start from a
  tractable 1D baseline rally (Approximation 1) as a testbed, with a roadmap
  toward a more realistic 2D court, net physics, and multi-phase
  (dink/attack/volley) rallies

## Formalization

### Environment: 1D Baseline Rally (Approximation 1)

- 1D court: positions $x \in [0, L]$, where $L$ is the baseline length (e.g.,
  20 units)
- Turn-based play: player hits -> ball travels -> opponent receives
  - Ball motion uses a fixed travel time $t_{transit}$ based on distance, at
    constant speed
  - No intermediate trajectory simulation
- Rally ends when:
  - Opponent cannot reach the ball in time
  - Opponent commits an unforced error
  - Ball goes out

### Turn Sequence

1. Attacking player chooses `target_x` $\in [0, L]$
2. Traveling player has
   $t_{transit}(|x_{ball} - x_{player}|) = |x_{ball} - x_{player}| / v_{ball}$
   time to reach the ball
3. If travel time is sufficient ($t_{transit} \geq 0$):
   - Player moves toward the ball, reaching $x_{ball} + \text{noise}$ per
     precision
   - If reached in time: hit ball to opponent
   - Otherwise: miss (winner shot for the attacker)
4. If travel time is insufficient: opponent wins the point (forced error or
   winner)

### State Space

- Observation (fully observable per player):
  - `t`: step within rally (0 to max_rally_length)
  - `x_ball`: ball position, $x \in [0, L]$
  - `x_player`: own position
  - `x_opponent`: opponent position
  - `whose_turn`: whose turn to hit
- Derived:
  - `time_to_reach = |x_ball - x_player| / v_ball`
  - `ball_reachable = (time_to_reach <= t_move(distance, speed))`

### Action Space

- Discrete (preferred for simplicity): 5 actions
  `[aim_left, aim_mid_left, aim_center, aim_mid_right, aim_right]`
  - Maps to target positions $x_{target} \in \{0.2L, 0.4L, 0.5L, 0.6L, 0.8L\}$
  - Shot-type variant (Option B): 10 actions = 5 positions x 2 shot types
    (forehand vs backhand, depending on player position relative to ball)
- Continuous (for future work):
  $a = (x_{target}, v_{spin}) \in [0,L] \times [0, v_{max}]$, adding depth
  control (net clearance, pace)
- Execution noise:
  - Intended $x_{target}$ maps to realized
    $x_{hit} = x_{target} + \mathcal{N}(0, \sigma_{precision}^2)$
  - $\sigma_{precision}$ is player-specific: higher skill implies lower noise
  - Ball is out if $|x_{hit}| > L$, with probability proportional to distance
    from target and skill

### Player Models

- Option A: aggregate skill (simpler)
  - Parameters: movement speed $v_{move}$, shot precision
    $\sigma_{precision}$, unforced error rate $p_{ufoe}$, reaction time
    $t_{react}$
  - Higher skill implies lower $p_{ufoe}$
  - Defined at 3-5 discrete skill levels (e.g., beginner, intermediate,
    advanced)
- Option B: shot-type breakdown (advanced)
  - Separate parameters per shot: forehand ($\sigma_{FH}$, $p_{FH,ufoe}$),
    backhand ($\sigma_{BH}$, $p_{BH,ufoe}$)
  - Weak-side penalty: backhand is inherently worse ($\sigma_{BH} >
    \sigma_{FH}$, $p_{BH,ufoe} > p_{FH,ufoe}$)
  - Strategy implication: agent should learn to force the opponent to the
    weak (backhand) side

### Reward Function

- Per-step: $r_t = 0$ while the rally is ongoing (no reward until
  termination)
- Terminal reward: $r = +1$ for a win, $r = -1$ for a loss
- Optional step cost: $-0.001 \cdot |x_{target} - x_{current}|$ penalizes
  large position changes, pushing toward economical strategy (ablate with and
  without)
- Alternative metric: raw win rate vs baseline, tracked separately from the
  reward signal

### Assumptions (Approximation 1)

- Perfect information: both players observe the full state (relaxation:
  partial observability, reaction delay)
- Fixed ball speed: no pace variation (relaxation: continuous spin/speed
  action)
- 1D only: no net, no court depth (relaxation: 2D court in Approximation 2)
- Fixed single-opponent skill: no opponent learning during play (relaxation:
  adaptive opponent or multi-agent training)

## Key Examples

- **Exploiting the weak side**: under Option B, an agent facing an opponent
  with a weaker backhand ($\sigma_{BH} > \sigma_{FH}$) directs a
  disproportionate share of shots to the backhand side, mirroring the
  coaching principle "play the backhand"
- **Forced winner via travel-time deficit**: attacker targets $x_{target} =
  0.8L$ while the opponent stands near $0.2L$; if $t_{transit}$ exceeds the
  opponent's move budget, the point ends in a winner with no trajectory
  simulation needed
- **Symmetric-skill sanity check**: when both players share identical
  parameters ($v_{move}$, $\sigma_{precision}$, $p_{ufoe}$), the trained
  self-play policy should converge to roughly a 50% win rate; this edge case
  validates the environment implementation rather than genuine strategic
  skill

## Questions

1. Does a policy trained on the 1D Approximation 1 transfer qualitatively to
   the 2D Approximation 2 court, or does it need to be retrained largely from
   scratch?
2. What quantitative test would confirm that a learned strategy matches real
   coaching heuristics (e.g., backhand exploitation), beyond visually
   inspecting shot distributions and position heatmaps?
3. If the learned policy converges to a fixed exploitative pattern (e.g.,
   always targeting the backhand), does it generalize to opponents with
   different weaknesses, or is it overfit to the fixed baseline opponents
   used during training?

## Research Topics

- **Training algorithm**: DQN (discrete actions) vs PPO (continuous actions)
  trade-offs for this rally formulation, and self-play stability
- **Reward shaping**: terminal win/loss reward only vs an added step-cost
  penalty on large position changes, and its effect on qualitative strategy
  (economical vs aggressive movement)
- **Strategy validation methodology**: quantitative tests, beyond visually
  inspecting shot distributions and heatmaps, for confirming a learned policy
  matches real coaching heuristics
- **Roadmap beyond Approximation 2**:
  - Approximation 3: multi-phase rally model (dinking phase and attack phase)
  - Approximation 4: continuous ball speed/spin actions for pace variation

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: build the 1D rally simulator and Gym-compatible environment
  - Implement the turn-based game engine (turn sequence, error/out detection)
  - Implement the Option A (aggregate skill) player model with 3-5 discrete
    skill levels
  - Wrap it as a Gym-compatible RL environment (state/action/reward as in
    Formalization)
  - Implement baselines: random policy, aim-opposite-to-opponent heuristic,
    and a minimax / backward-induction optimal solver
  - This is the result: a tested 1D rally environment with baseline agents to
    compare against

- Milestone 2: train and validate a self-play RL agent
  - Train a DQN or PPO agent via self-play, snapshotting periodically
  - Run 10k-100k episodes, tracking win rate, rally length, and action
    distribution
  - This is the result: an agent reaching a >90% win rate vs the random
    baseline, with learning curves confirming convergence

- Milestone 3: validate learned strategy against coaching heuristics
  - Add the Option B (shot-type breakdown) player model with a weak-side
    (backhand) penalty
  - Run the skill-level ablation (train vs beginner/intermediate/advanced)
    and measure exploit-asymmetry (share of shots to the backhand)
  - This is the result: quantitative evidence for or against the agent
    learning to exploit the opponent's weak side and adapt to opponent skill

- Milestone 4: extend to Approximation 2 (2D court) and test strategy
  transfer
  - Build the 2D court environment with net clearance
  - Benchmark the 1D-trained policy directly on the 2D environment, then
    fine-tune it
  - This is the result: measured drop in win rate on direct transfer, and a
    comparison of fine-tuning convergence speed against training from
    scratch

## References

- TBD
