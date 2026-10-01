# Implementing Monte Carlo Tree Search and AlphaZero

## Status
- **Status:**: in_progress
- **Complete Specs:**: 20%

## Core Idea

- Create an accessible, step-by-step tutorial implementing Monte Carlo Tree
  Search (MCTS) and AlphaZero from scratch
- Goal: make these game-playing algorithms understandable through hands-on
  Python code, starting with simple games (Tic-Tac-Toe) and progressing to
  more complex domains (Chess, Go)
- This addresses the gap between theoretical papers and practical
  implementation, letting students and practitioners understand the
  interplay between tree search, neural networks, and self-play learning
- Core insight: AlphaZero's power comes from combining three ideas: MCTS for
  efficient exploration, neural network guidance, and self-play iteration,
  each learnable independently and then integrated
- Breaking this into digestible pieces makes the algorithm both
  understandable and implementable

## Formalization

- AlphaZero combines three components:
  - **Tree search**: MCTS balances exploration vs. exploitation using Upper
    Confidence bounds applied to Trees (UCT)
    ```text
    UCT(node) = Q(node)/N(node) + C * sqrt(ln(N(parent))/N(node))
    ```
  - **Neural network**: policy and value heads trained on self-play
    ```text
    p, v = network(state)
    p: probability distribution over actions
    v: scalar value estimate for the position
    ```
  - **Self-play update**: each iteration improves the network by playing
    against itself
    ```text
    For each game:
      - Use MCTS + current network to generate moves
      - Collect (state, action, result) tuples
      - Retrain network on collected data
    ```

## Key Examples

- **Tic-Tac-Toe with pure MCTS**: shows tree search working without neural
  networks, playable in seconds
- **Tic-Tac-Toe with AlphaZero**: shows how adding a small network and
  self-play improves performance and learning speed
- **Connect Four or a simple chess variant**: demonstrates scaling
  challenges, and how network size and training time matter

## Questions

1. How do we balance MCTS simulation count vs. network quality in early
   training?
2. What is the minimum network architecture needed to beat strong MCTS-only
   baselines?
3. How does the curriculum (game complexity progression) affect learning
   speed?
4. Can we extract interpretable insights about optimal game strategy from
   trained networks?

## Research Topics

- **MCTS variants**: parallel MCTS, progressive widening, rave (rapid action
  value estimation)
- **Network architecture**: how shallow can networks be, and the effect of
  network width on sample efficiency
- **Self-play curriculum**: temperature in move selection, training data
  retention, online vs. offline learning
- **Game abstractions**: can networks learn generalizable patterns across
  similar games?

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., Tic-Tac-Toe MCTS in 1-2 days)
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at a time

## Implementation plan

- Milestone 1: pure MCTS (Tic-Tac-Toe)
  - Implement game state, rules, and the basic MCTS algorithm
  - This is the result: a playable, unbeaten MCTS player

- Milestone 2: neural network integration
  - Add a small conv/dense network for policy and value prediction
  - Integrate network guidance into the MCTS tree
  - This is the result: an MCTS and network baseline

- Milestone 3: self-play training
  - Implement the self-play loop and network training
  - Add move generation and data collection
  - This is the result: a trained AlphaZero agent playing against itself
    with improving performance

- Milestone 4: scaling and analysis
  - Extend to Connect Four or chess variants
  - Analyze learned representations and strategy patterns
  - This is the result: a reproducible, generalizable implementation

## References

- Silver, D., et al. _Mastering the Game of Go with Deep Neural Networks and
  Tree Search_. Nature (2016)
- Silver, D., et al. _Mastering Chess and Shogi by Self-Play with a General
  Reinforcement Learning Algorithm_. arXiv (2017)
- Browne, C., et al. _A Survey of Monte Carlo Tree Search Methods_. IEEE
  Transactions on Computational Intelligence and AI in Games (2012)
- Surag Nair. _AlphaZero General_. GitHub repository
  (`suragnair/alpha-zero-general`) with a comprehensive tutorial and sample
  implementations (Othello, GoBang, TicTacToe) in PyTorch and Keras
- FreeCodeCamp. _AlphaZero from Scratch_. 4-5 hour YouTube tutorial covering
  TicTacToe, neural networks, self-play, training, and Connect Four, with
  code and trained models
- IEEE Conference tutorials on MCTS: history and relationship to
  simulation-based algorithms for Markov decision processes, with
  tic-tac-toe and AlphaGo/AlphaZero demonstrations
- INFORMS tutorial on MCTS fundamentals, with decision tree and tic-tac-toe
  demonstrations, available at informs-sim.org
- `ai-boson.github.io/mcts/`: beginner-friendly Python tutorial on MCTS
  algorithm design, covering Go, Sudoku, Tic Tac Toe, and Chess
- JuliaCon 2021 talk: a ten-minute overview of AlphaZero fundamentals
- Medium by Darin Straus: AlphaZero implementation details in Python with
  TensorFlow, with code on GitHub
- Kaggle tutorial, _AlphaZero from Scratch_: theory and references
- https://www.dwarkesh.com/p/eric-jang
- https://evjang.com/2026/04/28/autogo.html#cover
