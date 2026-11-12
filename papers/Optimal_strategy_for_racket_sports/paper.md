---
title: "Optimal Shot Placement in Racket Sports: A Game-Theoretic Framework"
author:
  - name: "`Pranav Shashidhara, Giacinto Paolo (GP) Saggese, Aahan Rembersu`{=typst}"
    department: "Department of Computer Science"
    organization: "University of Maryland, College Park"
    location: "College Park, MD, USA"
    email: "`pshashid@umd.edu, gsaggese@umd.edu, arembers@terpmail.umd.edu`{=typst}"
abstract: |
  Where to place a shot is a central tactical decision in tennis and pickleball.
  Full three-dimensional simulation is expensive and hard to reason about, and
  tracking data describe what players do, not what they should do.
  <!-- -->
  This paper models the return of serve in two dimensions. The return lands at a
  Gaussian point around the aim point, and it is a winner if it lands in and the
  server, after a reaction delay, cannot run to it before it lands. We estimate
  the winner probability of every aim point on a grid by Monte Carlo, as one
  joint event on common random numbers, and report the best placement as a
  region of statistically tied aim points.
  <!-- -->
  A natural intuition is to aim near the lines and corners. At the default
  speeds of the model, the best return is instead the middle of the server's
  half in both sports: landing in matters more than landing far from the
  opponent. Placement starts to matter only with a large error spread, or with a
  slow ball and a fast server, and with no reaction time the two speeds act only
  through their ratio. A one-step reinforcement learning environment built on
  the same model reproduces the winner probability map from the reward alone,
  and a PPO agent trained on it also aims at the middle.
  <!-- -->
  We also describe the general framework that this model instantiates: a
  trajectory model, a grid score with reachability, and a zero-sum game over
  placements.
keywords:
  - tennis
  - pickleball
  - shot placement
  - game theory
  - trajectory optimization
  - Monte Carlo methods
bibliography: references.bib
---

# Introduction

Deciding where to send the ball is, together with deciding how to hit it,
the fundamental tactical choice in racket sports. A shot that lands safely
in the court but travels straight to the opponent achieves little; a shot
aimed at the extreme corner of the court is more likely to be a winner if it
lands in, but is also more likely to land out.

Optimal shot placement is therefore a decision under uncertainty, made under
tight time constraints, and made against an opponent who reacts strategically to
the pattern of one's choices.

Prior work on this problem tends to fall into two categories.

The first category relies on high-fidelity physical simulation of ball flight,
typically in three dimensions and including spin and aerodynamic drag, combined
with Monte Carlo sampling over the space of possible strokes [@cross2014drag].
Such simulations are accurate but computationally expensive, and their complexity
makes it difficult to reason analytically about why one placement is preferable
to another.

The second category is data-driven: it uses ball- and player-tracking data to
predict where a shot is likely to go [@wei2013shotlocations] or to evaluate the
outcomes of shots that were actually played [@kovalchik2020voncramm]. These
methods are descriptive rather than prescriptive; they characterize observed
behavior but do not, by themselves, prescribe an optimal placement policy from
first principles.

This paper proposes a middle ground: a reduced-order, first-principles framework
that is simple enough to reason about analytically, yet rich enough to capture
the trade-off between shot accuracy and opponent reachability that we regard as
the crux of shot-placement strategy.

We compute with its simplest instance, a two-dimensional model of the return of
serve, which rests on three simplifications.

- The flight of the ball is not simulated. The return lands at a random point,
  drawn from a Gaussian around the aim point, and flies to it in a straight line
  at a constant speed. The trajectory model of Section III-C is one way to derive
  this spread from the errors of the stroke.
- The court is discretized into a grid of aim points, and the probability that
  a return aimed at each one is a winner is estimated by Monte Carlo, with the
  same random draws for every aim point.
- The server reaches the ball if, after a fixed reaction time, they can run to
  the landing point at a constant top speed before the ball lands. This is
  checked for each sampled landing point, not as a separate probability.

The headline result runs against a natural intuition, which is to aim a return
near the lines and corners, far from the opponent.
<!-- TODO(GP): keep this as an intuition, or cite evidence of where players aim returns (P16)? -->
At the default speeds of the model, whose reaction time and player speed are
within the ranges reported in the literature (Section III-B), the best return is
instead the middle of the server's half, in both sports ([@fig-winner-map]): a
return near a line is more likely to land out, and the server covers too little
ground before the ball lands for distance from them to pay for that risk. A
reinforcement learning agent that sees only whether each return was a winner
learns the same answer (Section VIII).

The main contributions of this paper are:

- A two-dimensional model of the return of serve, with a closed form for the
  in-bounds probability and a Monte Carlo estimate of the winner probability as
  one joint event, on common random numbers, with the best placement reported
  as a region of statistically tied aim points (Sections IV-G and IV-H).
- A study of which variables change the answer, with a proposition that, with
  no reaction time, the ball and player speeds act only through their ratio
  (Section VII-A and Appendix A).
- A one-step reinforcement learning environment built on the same model, which
  reproduces the winner probability map from the reward alone, and a PPO agent
  evaluated on it with a pass rule written before the run (Section VIII).
- The general framework that the 2D model instantiates: a 1D/2D reduction of
  the trajectory-feasibility problem with an error-propagation model (Section
  III), a grid score that combines in-bounds probability with a deterministic
  reachability indicator (Sections IV-A to IV-F), and a zero-sum game over
  placements connected to empirical evidence of minimax play at the
  professional level [@walker2001minimax] (Section V).
- A demonstration that tennis and pickleball can be described by the same
  parametrized model, differing primarily in maximum ball speed and court
  scale, illustrated with a small worked example that exposes qualitatively
  different optimal regimes for the two sports (Sections VI and VII).

The paper is organized as follows. Section II reviews related work. Section
III formalizes the problem. Section IV describes the grid-based scoring
methodology and, in Sections IV-G and IV-H, the 2D model we compute with.
Section V develops the game-theoretic layer. Section VI works a small
illustrative example by hand. Section VII discusses the tennis and pickleball
parametrizations and which variables matter. Section VIII learns the return of
serve with reinforcement learning. Section IX discusses limitations, and Section
X concludes. Appendix B lists, for each result, the notebook that reproduces it
and the entry of the project's research log that holds its runs and figures.

# Related Work

**Ball trajectory physics.** The aerodynamics of racket-sport balls, including
drag and Magnus-effect lift due to spin, have been measured directly using
high-speed cameras and wind-tunnel experiments [@cross2014drag]. These
measurements show that drag and lift forces are not negligible relative to
gravity, particularly at the higher speeds reached by serves and hard
groundstrokes. Our framework deliberately ignores these effects to keep the
trajectory problem analytically tractable, at the cost of introducing a
systematic bias that we return to in Section IX.

**Data-driven shot analytics.** Using Hawk-Eye-style tracking data, prior
work has built probabilistic models that predict the location of a
player's next shot from the spatiotemporal features of the current rally
[@wei2013shotlocations], and has modeled the space of plausible future shot
trajectories to evaluate the quality of a player's decision relative to
the alternatives available at the time [@kovalchik2020voncramm]. These
methods answer "what shot is this player likely to play, and how good was
it," which is a different question from "what shot should any player play
given the current geometry," the question addressed here. Our framework
does not require historical tracking data and can, in principle, be
evaluated for any court geometry and any pair of player positions.

**Game theory in racket sports.** The clearest empirical link between game
theory and tennis strategy is the finding that the serve-direction choices
of elite players are statistically consistent with mixed-strategy minimax
equilibrium: players who lose a disproportionate share of points on serves
to one side subsequently increase the frequency of serves to the other
side, and the win probabilities across serve directions are close to equal,
as predicted by minimax play [@walker2001minimax]. That result concerns a
small, discrete choice, direction of serve, left, center, or wide. Section V
extends the same equilibrium logic to a continuous grid of target cells and
to shots beyond the serve.

**Reinforcement learning.** We use reinforcement learning not to find a new
answer, but as an independent check of the model: an agent that only plays
points should find the same best return as the Monte Carlo maps. A single return
of serve is a contextual bandit, a one-step decision with a context, the serve,
and a reward [@lattimore2020bandit]. We state it as a Gymnasium environment
[@towers2024gymnasium] and train the PPO algorithm [@schulman2017ppo], as
implemented in Stable-Baselines3 [@raffin2021sb3]. Extending the environment to
rallies and to an opponent who adapts leads back to the game of Section V.

# Problem Formulation

## Court Geometry and Notation

We model the court as a planar rectangle and place the origin at the
striker's contact point.

Table I summarizes the court dimensions and net heights defined by the respective
governing bodies, together with the fastest recorded shot speed in each sport,
which we use as an upper bound on the maximum ball speed parameter $v_{\max}$.

: Court and ball-speed parameters for tennis and pickleball

| Parameter | Tennis | Pickleball |
| :--- | ---: | ---: |
| Court (singles), length x width | 23.77 m x 8.23 m | 13.41 m x 6.10 m |
| Net height, center / post | 0.914 m / 1.07 m | 0.86 m / 0.91 m |
| Non-volley zone (from net) | n/a | 2.13 m |
| Fastest recorded shot | 263.4 km/h (73.2 m/s) | 99.8 km/h (27.7 m/s) |

Tennis dimensions and net heights are from the ITF Rules of Tennis
[@itf2024rules]; pickleball dimensions are from the USA Pickleball official
rulebook [@usap2024rulebook]; the fastest-shot figures are the certified
records for a men's serve [@guinness2012serve] and for a hard-hit shot in
pickleball [@guinness2025pickleball], respectively.

## Simplifying Assumptions

The framework rests on the following approximations, which trade physical
fidelity for tractability:

- **Point-mass ball and point players.** Racket or paddle contact is
  modeled as an instantaneous velocity change at a point, and players are
  modeled as points on the court plane rather than as extended bodies with
  reach.
- **No air drag or spin.** Ball flight follows unpowered projectile motion.
  This is the most consequential simplification; Section IX discusses
  its likely impact using measured drag and lift coefficients
  [@cross2014drag].
- **Bounded reaction time.** A player requires a fixed latency after ball contact
  before beginning to move. We use the 190-250 ms range reported for simple
  visual reaction time as a plausible baseline [@woods2015reaction], noting that
  this can be a conservative estimate: anticipation and cue reading can shorten
  the effective decision latency in practice, while decision complexity can
  lengthen it.
- **Bounded movement speed.** A player's on-court movement speed is bounded
  by a sport- and player-specific maximum $v_p$. Measured average movement
  speeds immediately before and after the split-step range from about 0.55
  to 1.9 m/s depending on stroke type and player level
  [@filipcic2017splitstep]; peak short-burst court-coverage speed is
  higher, and we treat $v_p$ as a free parameter of the model rather than
  fixing it to a single literature value.
- **Groundstrokes only.** We model rallies in which each shot bounces
  before the opponent responds, and exclude volleys, in which a player
  strikes the ball before it bounces. Extending the framework to volleys is
  left to future work.

## Problem 1: Trajectory Feasibility

For a given shot, let:

- $S$ denote the striker's contact point
- $N$ the point where the trajectory crosses the net, and
- $T$ a candidate target on the opponent's side of the court.

Because $S$, $N$, and $T$ are, by construction, coplanar for a fixed aim
direction, we work in the vertical plane containing the line $ST$, with
horizontal coordinate $x$ (distance along $ST$) and vertical coordinate $z$
(height above the court surface). Lateral placement, i.e., the choice of aim
direction itself, is treated as a separate, second angular degree of freedom
(Section III-C).

Given a contact point $S$ at height $h_0$, a net at horizontal distance
$d_{\mathrm{net}}$ and height $h_{\mathrm{net}}$, and a target $T$ at
horizontal distance $d_T$, the trajectory-feasibility problem asks for a
launch speed $v_0$ and launch angle $\theta$ (measured from horizontal)
such that the ball clears the net and lands at $T$. Under the no-drag
assumption, the trajectory in the vertical plane is

$$
x(t) = v_0 \cos\theta \, t, \qquad
z(t) = h_0 + v_0 \sin\theta \, t - \frac{1}{2} g t^2 .
$$

The flight time to landing ($z = 0$) is

$$
T_f = \frac{v_0 \sin\theta + \sqrt{(v_0 \sin\theta)^2 + 2 g h_0}}{g} ,
$$

and the net-clearance constraint requires
$z(t_{\mathrm{net}}) > h_{\mathrm{net}}$ at
$t_{\mathrm{net}} = d_{\mathrm{net}} / (v_0 \cos\theta)$. The feasible set
$\mathcal{F}(T)$ for hitting $T$ within a tolerance $\epsilon$ then consists
of every pair $(v_0, \theta)$ that satisfies both the net-clearance
condition above and the landing condition

$$
x(T_f) \in [\, d_T - \epsilon, \; d_T + \epsilon \,] ,
$$

subject to $v_0 \le v_{\max}$ for the sport in question (Table I). Because this
is a two-parameter feasibility problem posed along a single spatial axis, we
refer to it as the "1D" trajectory problem, in contrast to a full 3D flight
simulation.

A player cannot execute $(v_0, \theta)$ exactly; every stroke carries
execution noise. We model this noise as

$$
\theta = \bar\theta + \delta_\theta, \qquad \delta_\theta \sim
\mathcal{N}(0, \sigma_\theta^2),
$$

$$
v_0 = \bar v_0 (1 + \delta_v), \qquad \delta_v \sim \mathcal{N}(0,
\sigma_v^2),
$$

where $\bar\theta$ and $\bar v_0$ are the intended (aim-point-feasible)
angle and speed. A third error term, the lateral angle $\phi \sim
\mathcal{N}(0, \sigma_\phi^2)$, perturbs the aim direction within the
horizontal plane and is what makes the landing distribution genuinely
two-dimensional rather than confined to the line $ST$. Propagating
$(\delta_\theta, \delta_v, \phi)$ through the trajectory equations, either
by direct Monte Carlo sampling or by local linearization for small errors,
yields a distribution over landing points on the court plane, which we use
in Section IV to score candidate targets.

## Problem 2: Placement Selection

Given the current ball state at the moment of contact at the attacking
player's position $S$ and the returning opponent's court position $O$, the
placement-selection problem asks for the target $T$ that maximizes the
chance the shot is both a legal, in-bounds shot and one the opponent
cannot reach before it bounces twice. This requires combining the landing
distribution from Section III-C with a model of the opponent's response
capability, which we formalize next.

The computations of this paper use a two-dimensional reduction of both
problems, defined in Sections IV-G and IV-H. The landing point is drawn directly
from a Gaussian around the aim point, instead of propagating stroke errors
through the trajectory of Problem 1, and the opponent's reach is checked for each
sampled landing point. Sections VII and VIII study this model.

# Grid-Based Reachability and Monte Carlo Scoring

## Court Discretization

We discretize the opponent's court into a grid of candidate target cells
$\mathcal{C} = \{c_1, \dots, c_M\}$. Grid resolution trades accuracy for
computational cost: a finer grid better approximates the continuous
placement decision but requires proportionally more Monte Carlo samples
and more reachability evaluations, discussed in Section IV-F.

## Landing Probability via Monte Carlo

For a candidate cell $c$ with center $T_c$, we compute the
feasibility-constrained aim parameters $(\bar\theta, \bar v_0)$ from
Section III-C, draw $K$ samples of $(\delta_\theta, \delta_v, \phi)$ from
the error model, simulate the resulting trajectory for each sample, and
estimate

$$
\begin{aligned}
P_{\mathrm{in}}(c) \approx \frac{1}{K} \sum_{k=1}^{K} \mathbb{1}[\,
& z_k(t_{\mathrm{net}}) > h_{\mathrm{net}} \\
& \wedge \mathrm{land}(\theta_k, v_{0,k}, \phi_k) \in \mathcal{L} \,] ,
\end{aligned}
$$

where $\mathcal{L}$ is the legal landing region (the opponent's half in a
rally, the service box for a serve) and $z_k(t_{\mathrm{net}}) >
h_{\mathrm{net}}$ is the net-clearance condition of Section III-C for sample
$k$. A sample counts if the ball clears the net and lands anywhere in
$\mathcal{L}$: the cell $c$ is where the striker aims, not where the ball must
land, so $P_{\mathrm{in}}(c)$ is the in-bounds probability of a shot aimed at
$c$. The feasible set $\mathcal{F}(T_c)$ holds many launch pairs: we evaluate a
grid of feasible launch angles $\bar\theta$ for each cell, each with the launch
speed that lands it at $T_c$, and keep for each cell the angle with the highest
score $S(c)$ of Section IV-D, since a higher, slower shot flies longer and the
best angle depends on the opponent's position.

Because the underlying trajectory model is one-dimensional per sample
(Section III-C), each Monte Carlo draw costs a closed-form evaluation
rather than a numerical integration of drag-augmented equations of motion,
which is what makes $K$ in the thousands practical per cell even for
moderately fine grids.

## Deterministic Reachability

Let $t_r$ denote the opponent's reaction time and $v_p$ the opponent's
maximum movement speed (Section III-B). For a candidate cell $c$ at
distance $d(O, c)$ from the opponent's current position $O$, we define
reachability as the deterministic indicator

$$
R(c) = \mathbb{1}\big[\, t_r + d(O,c)/v_p \; \le \; T_f(c) \,\big] ,
$$

where $T_f(c)$ is the ball's flight time to $c$ from Section III-C. Unlike
$P_{\mathrm{in}}$, $R$ is not estimated by sampling: given fixed $t_r$,
$v_p$, and $T_f(c)$, whether the opponent's minimum possible arrival time
is within the flight-time budget is a yes/no geometric fact, not a random
outcome. Randomness in the opponent's true reaction time or top speed can
be incorporated by treating $R(c)$ as a probability rather than an
indicator; we retain the deterministic form here for tractability and note
the extension in Section IX.

## Composite Score and Candidate Selection

We combine the two quantities into a single score per cell,

$$
S(c) = P_{\mathrm{in}}(c) \cdot \big(1 - R(c)\big) ,
$$

which is zero for any cell the opponent can reach, regardless of how safe
the shot is, and equal to the in-bounds probability for any cell the
opponent cannot reach. The candidate placement is then

$$
c^\star = \arg\max_{c \in \mathcal{C}} S(c) .
$$

This multiplicative form directly encodes the qualitative intuition that a
placement is only as good as the product of "does it land in" and "can the
opponent not get there." Section VI works through a concrete instance of
this scoring rule. Weighted or non-multiplicative combinations of
$P_{\mathrm{in}}$ and $R$ are possible, for example to express risk
aversion, but are left to future work.

## Serve and Return

The same grid-and-score procedure applies to serving and to returning
serve, with two changes to the setup: the target grid is restricted to the
legal service box, and $v_{\max}$ and the tolerance $\epsilon$ reflect the
serve motion rather than a groundstroke. The opponent's position $O$ for a
return is the returner's ready position rather than a mid-rally recovery
position, and $t_r$, $v_p$ can differ from rally-play values because return
positioning and split-step timing are typically more standardized than
in-rally movement.

## Computational Complexity

Evaluating one grid consists of $M$ cells, each requiring $K$ trajectory
samples for $P_{\mathrm{in}}$ and one closed-form reachability check for
$R$, giving $O(MK)$ closed-form trajectory evaluations per decision. This
is substantially cheaper than a full 3D drag-and-spin simulation
integrated numerically over the same $M \times K$ grid, because each
sample here is an $O(1)$ evaluation of the closed-form no-drag equations of
Section III-C rather than a numerical ODE solve. The reduction from 3D to a
per-cell 1D/2D trajectory model is therefore what makes exhaustive grid
scoring, rather than a small number of hand-picked candidate shots,
computationally practical within the time constraints of a rally.

## Closed-Form In-Bounds Probability for a Gaussian Landing Point

Section IV-B estimates $P_{\mathrm{in}}$ by propagating stroke errors through
the trajectory model. The model we compute with is simpler: it places the
uncertainty directly on the landing point. A shot aimed at $(x_2, y_2)$ lands at
$(X, Y)$ with

$$
X \sim \mathcal{N}(x_2, \sigma_x^2), \qquad Y \sim \mathcal{N}(y_2,
\sigma_y^2),
$$

where $X$ and $Y$ are independent. This model uses a frame with the origin at
the center of the net, $x$ across the court and $y$ along it, rather than the
striker-centered frame of Section III-A. For a court of length $L$ and width
$W$, a shot toward the half $y \le 0$ is in when $|X| \le W/2$ and
$-L/2 \le Y \le 0$. Because the two coordinates are independent and this
region is a rectangle, the event splits into one condition on $X$ and one on
$Y$, and $P_{\mathrm{in}}$ is the product of two differences of the standard
normal CDF $\Phi$:

$$
\begin{aligned}
P_{\mathrm{in}} &= p_x \, p_y , \\
p_x &= \Phi\left(\frac{W/2 - x_2}{\sigma_x}\right)
- \Phi\left(\frac{-W/2 - x_2}{\sigma_x}\right) , \\
p_y &= \Phi\left(\frac{-y_2}{\sigma_y}\right)
- \Phi\left(\frac{-L/2 - y_2}{\sigma_y}\right) ,
\end{aligned}
$$

where $p_x$ is the probability that $X$ lands between the sidelines and $p_y$
the probability that $Y$ lands between the baseline and the net.

With correlated errors this would require a bivariate normal CDF, and with a
non-rectangular region a two-dimensional integral. The model ignores the net:
a shot counts as out only if it lands outside the target half.

In the accompanying notebook, the serve and return setting (compare Section
IV-E) is modeled as follows. The server chooses the deuce or the ad side and
stands behind the baseline on that side's half of the court. Only the service
box diagonally across the court is legal, and lines are in, except the
pickleball non-volley-zone line. A serve that is wide, lands in the wrong box,
lands long, or, in pickleball, lands in the non-volley zone is a fault, and a
fault ends the point with no return. After a legal serve to $(x_1, y_1)$, the
returner moves to $(x_1, y_1)$ and returns toward $(x_2, y_2)$ with spread
$(\sigma_x, \sigma_y)$, and $P_{\mathrm{in}}$ is the probability that the
return lands in the server's half. The closed form is checked against a Monte
Carlo estimate from samples of the same Gaussian, which agrees with it within
sampling error. It is implemented, with an interactive notebook for both
sports, in `notebooks/racket_sports_01_shot_probability_utils.py` of the
accompanying code.

Details: [Gdoc, Research results tab, Experiment 1](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0).

## Winner Probability of a Return

This subsection extends Section IV-G from
"does the return land in" to "is the return a winner": it lands in and the
server cannot reach it. After a legal serve to $(x_1, y_1)$, the returner hits
from $(x_1, y_1)$ toward $(x_2, y_2)$ with the Gaussian spread of Section IV-G,
and the server waits at the center of their baseline, $P = (0, -L/2)$. For
each of $K$ sampled landing points $(X_k, Y_k)$, the flight time is the
straight-line distance over a constant ground speed $v_b$ of the ball,

$$
T_k = \frac{\lVert (X_k, Y_k) - (x_1, y_1) \rVert}{v_b} ,
$$

so it depends on where the ball lands, not on where it was aimed. The server
reacts for $t_r$ and then runs in a straight line at speed $v_p$, and reaches
the landing point if

$$
\lVert (X_k, Y_k) - P \rVert \le v_p \max(0, \, T_k - t_r) .
$$

Unlike Section IV-C, the reach is evaluated for each sample. Let $W_k$ be the
event that sample $k$ lands in the server's half $H$ and is not reached. The
winner probability is estimated as this one joint event on the same samples,

$$
P_{\mathrm{win}}(x_2, y_2) \approx \frac{1}{K} \sum_{k=1}^{K}
\mathbb{1}[W_k] .
$$

It is not computed as the product
$P_{\mathrm{in}} \cdot P(\text{not reached})$, since both events depend on the
same landing point. The model is two-dimensional: the ball flies in a straight
line at a constant speed, the bounce is ignored, and the server must reach the
landing point by the time the ball lands.

$P_{\mathrm{win}}$ is evaluated at the centers of a grid over the server's half
($10 \times 15$ cells for tennis and $9 \times 10$ for pickleball, with
$K = 5000$), with common random numbers: every target uses the same standard
normal draws, shifted to its own aim point, so two targets differ because of
where they aim rather than because of sampling noise. Many targets can tie at
the top, for example when all $K$ samples are winners, so the best placement
is reported as a region: every target within two standard errors of the
highest $P_{\mathrm{win}}$, using the standard error of the difference of two
estimates, marked at the cell closest to the region's centroid. Each map is
identified by a settings string that lists all its inputs: the sport, the side,
the serve, the spread, $v_b$, $t_r$, $v_p$, the seed, the grid size, and $K$, so
a map reported in the research log can be rebuilt exactly in the notebook.

With a ball speed of 25 m/s for tennis, $t_r = 0.2$ s, and $v_p = 1.5$ m/s, a
return that lands next to the server flies about 15 m in 0.61 s, so the server
covers only about 0.6 m. $P_{\mathrm{win}}$ is then close to $P_{\mathrm{in}}$
except in a small dip around the server, and the best region is the middle of
the server's half rather than a spot near the lines ([@fig-winner-map]). The
region is large, 41 of 150 cells, because these cells tie within sampling
error near $P_{\mathrm{win}} = 1$: placement inside it hardly matters. Because the spread does
not depend on the ball speed in this model, a faster return never lowers
$P_{\mathrm{win}}$, so hitting as fast as possible always looks best; a speed
and accuracy trade-off is left to future work. The model is implemented, with
a heatmap of $P_{\mathrm{win}}$ for both sports, in
`notebooks/racket_sports_01_shot_probability_utils.py` of the accompanying
code.

![$P_{\mathrm{win}}$ of every aim point for tennis at the default settings: serve at $(-2.1, 3.2)$, spread $(0.5, 1.0)$ m, $v_b = 25$ m/s, $t_r = 0.2$ s, $v_p = 1.5$ m/s, and $K = 5000$. The black outline is the best region, within two standard errors of the highest value (41 of 150 cells), and the star is its middle: the best return is the middle of the server's half, not the lines or the corners.](figures/winner_heatmap_tennis.png){#fig-winner-map width=72%}

Details: [Gdoc, Research results tab, Experiment 2](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0).

# Game-Theoretic Placement

## Placement as a Zero-Sum Game

Section IV treats the opponent's position $O$ as fixed and known, and
selects the cell that maximizes $S(c)$ against that fixed position. A
rational opponent, however, anticipates the striker's tendencies and
adjusts $O$, the pre-shot recovery position, accordingly. If the striker
always aimed at the single highest-scoring cell, an opponent who learns
this would simply recover toward that cell, driving its effective $R(c)$
toward 1 and its score toward zero. Optimal play therefore requires mixing
across multiple candidate cells rather than deterministically selecting
the argmax.

We formalize repeated placement decisions as a zero-sum game in which the
striker chooses a mixed strategy $\sigma \in \Delta(\mathcal{C})$ over
target cells and the returner chooses a mixed strategy $\rho \in
\Delta(\mathcal{C})$ over anticipated recovery positions, with payoff
$u(c, c')$ to the striker equal to the point-win probability when the
shot is aimed at $c$ and the returner has recovered toward $c'$. The
equilibrium strategy is

$$
\sigma^\star = \arg\max_{\sigma \in \Delta(\mathcal{C})} \;
\min_{\rho \in \Delta(\mathcal{C})} \;
\mathbb{E}_{c \sim \sigma,\, c' \sim \rho}[\, u(c, c') \,] .
$$

For a grid with $M$ cells, this reduces to solving an $M \times M$
zero-sum matrix game, which is a linear program and is computationally far
cheaper than the $O(MK)$ scoring step that produces the payoff matrix in
the first place.

We do not solve this game in this paper. The computations of Sections VII and
VIII keep the server at a fixed position; extending them to a server who adapts,
for example by self-play, is future work (Section X).

## Connection to Minimax Play at Wimbledon

This formulation generalizes the discrete, three-direction (left, center,
wide) serve-placement game empirically validated by Walker and Wooders
[@walker2001minimax], who found that professional servers' direction
frequencies and the resulting win probabilities across directions are
statistically indistinguishable from minimax equilibrium play. Their result
concerns a fixed, small strategy space chosen by convention; the framework
proposed here suggests that the same equilibrium logic should, in
principle, extend to the full continuous target grid and to shots beyond
the serve, provided the payoff matrix $u(c, c')$ can be estimated. Testing
this generalization against real rally data is a natural direction for
future work (Section X).

## Post-Shot Positioning

A related and still-open question is where the striker should move
immediately after hitting the shot, before knowing the opponent's
response. This is the mirror image of the placement problem: having
committed to a distribution $\sigma$ over target cells, the
striker's own recovery position should maximize coverage against the
returner's best-response distribution over the striker's own court. In
general this defines a bilevel, repeated game in which each player's
optimal positioning depends on the other's strategy at the following
shot. We do not attempt to solve this repeated game in closed form here;
we note that it can, in principle, be approximated with the same grid
representation and standard zero-sum solution methods (e.g., the double
oracle algorithm), applied recursively shot by shot, and leave a full
treatment to future work.

# Illustrative Worked Example

To make the scoring rule of Section IV concrete, this section works
through a small, hand-computed example. It illustrates the scoring rule by
hand and is not a simulated experiment; the computed results use the 2D
model of Sections IV-G and IV-H and are in Sections VII and VIII.

## Setup

[@fig-trajectory] shows the trajectory-feasibility geometry underlying the example:
a contact point $S$ at height $h_0 = 1.0$ m, a net at $d_{\mathrm{net}} =
12$ m with $h_{\mathrm{net}} = 0.914$ m, and a target $T$ at $d_T = 20$ m.
For a launch angle $\theta = 8^\circ$ and speed $v_0 = 22.9$ m/s, the
resulting trajectory clears the net by about $0.40$ m and lands at $T$
after a flight time $T_f = 0.88$ s; the dashed curves show four perturbed
trajectories under $\pm 1.5^\circ$ and $\pm 3^\circ$ angle error combined
with $\mp 5$-$8\%$ speed error, and the shaded segment shows the resulting
spread of landing points.

![1D trajectory-feasibility geometry: contact point $S$, net, target $T$, the nominal trajectory (solid), four perturbed trajectories (dashed), and the resulting landing spread.](figures/trajectory_schematic.png){#fig-trajectory}

We consider five candidate target cells $c_1, \dots, c_5$ at increasing
lateral distance $d(O, c_i) \in \{0.3, 0.7, 1.1, 1.5, 1.9\}$ m from the
returner's recovery position $O$, with illustrative in-bounds
probabilities $P_{\mathrm{in}}(c_i) \in \{0.95, 0.91, 0.85, 0.77, 0.65\}$
decreasing with distance from center, consistent with the intuition that
more extreme targets carry greater execution risk. We use round,
illustrative values for the reachability parameters, chosen to be
consistent with the ranges discussed in Section III-B rather than fitted
to data: reaction time $t_r = 0.2$ s and movement speed $v_p = 1.5$ m/s.

## Two Regimes

**Tennis groundstroke exchange.** With $T_f = 0.8$ s (consistent with
[@fig-trajectory]), the available movement time after reacting is $T_f - t_r = 0.6$
s, giving a reach radius $v_p (T_f - t_r) = 0.9$ m. Cells $c_1$ and $c_2$
fall within this radius ($R = 1$); cells $c_3$-$c_5$ do not ($R = 0$).
Table II reports $P_{\mathrm{in}}$, $R$, and the composite score $S$ for
each cell; [@fig-toy] (left panel) shows the same values graphically. The
maximizing cell is $c_3$, with $S(c_3) = 0.85$: not the safest cell
($c_1$, which is reachable and therefore scores zero) and not the most
extreme cell ($c_5$, which is unreachable but carries the lowest
in-bounds probability), but the closest-to-center cell that is just
outside the opponent's reach envelope.

**Pickleball net exchange.** Pickleball rallies at the non-volley-zone line
involve much shorter ball-travel distances than tennis groundstrokes. As
an illustration, a fast exchange over a $5$ m distance at $25$ m/s, a firm
but sub-record pace relative to the certified $27.7$ m/s maximum in Table
I, gives $T_f = 0.2$ s. Because $T_f \le t_r$, the available movement time
is zero for every cell, so $R(c_i) = 0$ for all $i$ and $S(c) =
P_{\mathrm{in}}(c)$ everywhere (Table II, [@fig-toy], right panel). The
maximizing cell is simply the safest one, $c_1$. This qualitative
difference, reachability-driven trade-offs in tennis versus
accuracy-dominated selection in fast pickleball exchanges, arises entirely
from the ball-speed and distance parameters of Table I, with the scoring
mechanism of Section IV held fixed.

: Toy-example scores for the five candidate cells under the tennis and pickleball parametrizations. Bold marks the argmax cell in each regime.

| Cell | $d(O,c)$ (m) | $P_{\mathrm{in}}$ | $R$ (tennis) | $S$ (tennis) | $R$ (pball) | $S$ (pball) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| $c_1$ | 0.3 | 0.95 | 1 | 0.00 | 0 | **0.95** |
| $c_2$ | 0.7 | 0.91 | 1 | 0.00 | 0 | 0.91 |
| $c_3$ | 1.1 | 0.85 | 0 | **0.85** | 0 | 0.85 |
| $c_4$ | 1.5 | 0.77 | 0 | 0.77 | 0 | 0.77 |
| $c_5$ | 1.9 | 0.65 | 0 | 0.65 | 0 | 0.65 |

![Toy-example scores for five candidate target cells, tennis (left) versus a fast pickleball net exchange (right). Gray bars: in-bounds probability $P_{\mathrm{in}}$; colored bars: composite score $S$; star: argmax cell.](figures/toy_example_scores.png){#fig-toy}

# Tennis and Pickleball: A Unified Parametrization

The equations in Sections III-V make no sport-specific assumption beyond
the numeric values of $v_{\max}$, court dimensions, and net height
collected in Table I. Formally, both sports are instances of the same
model $\mathcal{M}(v_{\max}, \text{geometry}, \sigma_\theta, \sigma_v,
\sigma_\phi, t_r, v_p)$, and Section VI shows that the same scoring
mechanism, applied with pickleball-consistent parameters instead of
tennis-consistent ones, produces a qualitatively different optimal regime
without any change to the model structure. The practical differences
between the two sports, a smaller court, a lower net, a shorter or absent
non-volley approach zone, and a paddle rather than a strung racket, enter
the model exclusively through $v_{\max}$ and the geometric constants of
Table I, rather than through additional equations. This supports treating
"sport" as a parameter setting rather than a structural choice, as the
computations of Sections VII-A and VIII do.

## Which Variables Matter

The model of Section IV-H keeps $v_b$, $v_p$, $t_r$, and the spread
$(\sigma_x, \sigma_y)$ as separate variables. To see which of them change the
results, we measure each run by the highest $P_{\mathrm{win}}$ over the grid,
with the grid, the $K$, and the common random numbers of Section IV-H. The base
values are those of Section IV-H, with $v_b = 12$ m/s for pickleball and a
spread of $(0.5, 1.0)$ m. We first vary one variable at a time: $v_b$ from 5 m/s
to the model's maximum ball speed $v_{\max}$ (73.14 m/s for tennis and 24.59 m/s
for pickleball), $v_p$ from 0 to 6 m/s, $t_r$ from 0 to 1 s, and each spread from
0.1 to 3 m. We then cross nine serves with the ends and the base values of $v_b$
and $v_p$, for both service sides and both sports. The upper ends of $v_p$ and of
the spread bound the ranges of the model; they are not measured values. The
experiments are in `notebooks/racket_sports_05_variable_sensitivity.ipynb` of the
accompanying code.

The top of the $v_b$ range is the $v_{\max}$ of the sport parameters of the
accompanying code (`racket_params.py`), which is also the slider maximum of the
notebooks, so every map of the sweep can be rebuilt there. For tennis it matches
the record of Table I to within 0.1 m/s; for pickleball it is below the 27.7 m/s
record. The gap does not hide a drop: in this model a faster ball never lowers
$P_{\mathrm{win}}$ (Section IV-H), so a sweep up to the record could only raise
the highest $P_{\mathrm{win}}$ of pickleball, which is already 0.998 at
24.59 m/s.

**The error spread matters most.** With a spread of 3 m the highest
$P_{\mathrm{win}}$ falls to 0.827 for tennis and 0.695 for pickleball in $\sigma_x$,
and to 0.955 and 0.727 in $\sigma_y$ ([@fig-sweeps]). By contrast, $v_b$ and $v_p$ leave
tennis at 1.000 and pickleball between 0.948 and 0.998, and $t_r$ leaves pickleball
between 0.996 and 0.998.

![Highest $P_{\mathrm{win}}$ when one variable moves and the others stay at their base values.](figures/variable_sensitivity_sweeps.png){#fig-sweeps}

**Player speed and ball speed matter mainly in one corner.** The highest
$P_{\mathrm{win}}$ is at least 0.99 in 146 of the 162 tennis corner runs and in
108 of the 162 pickleball ones. The exception is a slow ball (5 m/s) with a fast
server (6 m/s), where it falls to between 0.816 and 0.991 for tennis and between
0.047 and 0.705 for pickleball, depending on the serve. There the serve matters,
and deep serves help the server ([@fig-serve-depth]): a deep serve pushes the returner
back, so the return flies farther and the server has more time to run. In
pickleball the short serves leave 0.680 to 0.705 and the deep ones 0.047 to
0.092, and the deep serve at the T is the lowest in both sports (on the tennis
ad side, tied with the other deep serves). Outside that corner, pickleball still
goes down to 0.904 at $v_p = 6$ m/s with the default ball. The mean of
$P_{\mathrm{win}}$ over the server's half moves more than its highest value, from
0.51 to 0.83 with $v_p$ in pickleball: the speeds change how much a poor aim costs
more than what the best aim can get.

![Highest $P_{\mathrm{win}}$ that the returner can get, by how deep the serve lands, for a slow ball and a fast server ($v_b = 5$ m/s, $v_p = 6$ m/s). A lower value is better for the server.](figures/variable_sensitivity_serve_depth.png){#fig-serve-depth}

**Reaction time changes little at realistic values.** Compared with $t_r = 0$,
a reaction time of 0.2 s at $v_p = 1.5$ m/s, within the ranges of Section III-B,
moves the best region by at most 0.26 m for tennis and 0.45 m for pickleball,
less than a grid cell, over ball speeds from 5 m/s to the fastest ball. The
highest $P_{\mathrm{win}}$ moves by 0.00 and 0.82 percentage points, and its mean
over the half by at most 2.34 and 3.95 points. A reaction time matters more for a
fast server, who loses the distance $v_p t_r$ while reacting: up to 17.52 points
on the highest $P_{\mathrm{win}}$ for pickleball at $v_p = 6$ m/s. It is not an
important variable at realistic values, and we keep it in the model for
completeness. Without it the two speeds are not separate variables:

**Proposition 1.** With $t_r = 0$, $P_{\mathrm{win}}$ of every target depends on
$v_p$ and $v_b$ only through the ratio $v_p / v_b$.

A proof sketch is in Appendix A. In simulation, multiplying both speeds by the
same factor gives identical maps, cell by cell, on 30 pairs of maps and on 500
random settings. With $t_r > 0$ the same scaling changes the maps, since a second
quantity enters, the reaction distance $v_p t_r$.

Two limits apply to these results. The highest $P_{\mathrm{win}}$ is 1.000 for many
tennis settings, so it cannot rank them. And the spread does not depend on the
ball speed in this model (Section IV-H), so a faster ball always looks better.

Details: [Gdoc, Research results tab, Experiment 3](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0)
(the sweeps and the serves) and
[Experiment 4](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0#heading=h.xvdn4n7zo860)
(the speed ratio and the reaction time).

# Learning the Return with Reinforcement Learning

Sections IV-H and VII-A compute the best return from the model. This section
asks whether an agent that only plays points, and sees only whether each return
was a winner, finds the same answer. It is the second of three phases of the
project: first the one-shot results without learning, then reinforcement
learning that reproduces them, and only then learning that goes beyond them, for
example to rallies or to a server who adapts. The agent reuses the code of the
model, the sampling of a landing point and the winner event, but not its
results: it never sees a map of $P_{\mathrm{win}}$, so agreement with Section
IV-H is evidence that the learning setup works, not a consequence of how it was
built.

## Environment

We cast one return of serve as a Gymnasium environment [@towers2024gymnasium]
in which the returner is the agent. One point is one shot, so an episode has a
single step, and the problem is a contextual bandit [@lattimore2020bandit]: the
serve is the context and the aim point is the arm.

- **Observation.** Six coordinates in meters: the server $(x, y)$, the returner
  $(x, y)$, and the serve landing $(x_1, y_1)$. In this version the server stays
  at the center of the baseline and the returner hits from the serve landing, as
  in Section IV-H, so only the serve varies; the two players are in the
  observation so that later versions can move them.
- **Reset.** A serve drawn uniformly over the deuce service box.
- **Action.** One cell of the grid of Section IV-H, $10 \times 15 = 150$ cells
  for tennis and $9 \times 10 = 90$ for pickleball. The return aims at the
  center of the cell.
- **Transition and reward.** One sample of the winner event $W_k$ of Section
  IV-H, drawn from the random generator of the environment, with the reward
  $r = \mathbb{1}[W]$. The episode then ends.

The spread, the ball speed, $t_r$, and $v_p$ are fixed at the values of Section
IV-H. Because the reward is the indicator of the winner event,

$$
\mathbb{E}[\, r \mid s, a \,] = P_{\mathrm{win}}(c_a \mid s) ,
$$

where $s$ is the serve and $c_a$ the cell of action $a$. The best action for a
serve is therefore the best target of Section IV-H, and the actions within
sampling error of it form its best region.

## Reference Answers

We compare the agent with two answers, each computed from scratch on the same
grid and for the same serves. The first is a brute force through the
environment: every action is played 1000 times, and the $k$-th play of every
action uses the same reset seed, so the actions share common random numbers as
in Section IV-H. The second is a fresh map of Section IV-H, with 5000 samples per
cell and a seed not used before. The environment reproduces Section IV-H if the
two maps have a Spearman rank correlation of at least 0.95, and the best cell of
each map lies in the best region of the other. The serves are 9 per sport, the
centers of a $3 \times 3$ grid over the deuce box.

## PPO Setup

The agent is PPO [@schulman2017ppo] as implemented in Stable-Baselines3
[@raffin2021sb3], version 2.9.2, with its default multilayer perceptron policy:
separate policy and value networks, each with two hidden layers of 64 tanh
units, and a softmax over the actions. The hyperparameters are the defaults of
the library: a learning rate of $3 \cdot 10^{-4}$, rollouts of 2048 steps, 10
epochs per update on minibatches of 64, a clipping range of 0.2, a value loss
coefficient of 0.5, a gradient norm clipped at 0.5, and no entropy bonus. With
one-step episodes the discount factor and the advantage estimator have no
effect, and the advantage of an action is its reward minus the value of the
serve.

Two choices matter. First, the observation is rescaled to $[-1, 1]$ with the
bounds of the court. The library feeds the raw observation to the network, and
in meters the constant coordinate of the server, $y = -11.885$ m in tennis,
saturated a quarter to a third of the first layer in early runs, which slowed
learning. Second, each run trains for 75,776 steps, 37 rollouts, on random
serves. On a reward that is almost always 1, PPO met the pass rule at about 75k
steps in development runs and drifted away from it after about 100k, so the
length was fixed before the final runs. The agent searches the whole grid, with
no action mask, no pruning with the results of Section IV-H, and no warm start.
We train one agent per sport for each of three seeds not used during
development.

## Pass Rule

The rule was written before the final runs, and a failing seed is reported as
it is, with no retuning. Let $p^\star(s) = \max_a P_{\mathrm{win}}(c_a \mid s)$
on the fresh map, and let $\pi(a \mid s)$ be the policy. We report:

- **Region.** Whether the greedy cell, $\arg\max_a \pi(a \mid s)$, lies in the
  best region of the fresh map. The strict rule asks this for every serve, seed,
  and sport.
- **Expected regret.** The winner probability that the policy loses on average,
  in percentage points:
  $$
  \mathrm{ER}(s) = \sum_a \pi(a \mid s) \,
  \big( p^\star(s) - P_{\mathrm{win}}(c_a \mid s) \big)
  $$
- **Greedy regret.** The same loss for the greedy cell alone.
- **Random baseline.** The expected regret of a uniformly random aim point,
  $p^\star(s)$ minus the mean of $P_{\mathrm{win}}$ over the cells.

The answer of Section IV-H is a region, not a point: many cells tie at the top,
most of all in tennis. We therefore lead with the expected regret and with the
region. A policy that spreads its probability over tied cells loses almost
nothing, even when its single most likely cell changes from seed to seed.

## Results

**The environment reproduces Section IV-H.** Over the 9 serves, the brute force
and the fresh map have a Spearman rank correlation of 0.980 to 0.982 for tennis
and 0.990 to 0.994 for pickleball, and the best cell of each map lies in the
best region of the other on 15 of 18 serves. The 3 misses are the short
pickleball serves, whose top cells tie within 0.2 points ([@fig-rl-maps]).

```{=typst}
#figure(
  image("figures/rl_env_vs_m2_pickleball.png", width: 100%),
  placement: top,
  scope: "parent",
  caption: [Pickleball, middle serve of the deuce box: P(winner) of every aim point from the brute force through the environment (left, 1000 points per cell) and from a fresh map of Section IV-H (right, 5000 samples per cell). Both put the best region, outlined in black, in the middle of the server's half behind the non-volley zone. Pickleball is the sharper test: its regions have 2 to 8 cells, against 40 to 45 of 150 in tennis. Player 1 is the server and player 2 the returner. The tennis maps are in the research log (Appendix B).],
) <fig-rl-maps>
```

**PPO aims at the middle.** On all six runs, three seeds per sport, the expected
regret is under 0.8 points for every serve, against 11 to 21 points for a random
aim point. The greedy cell lies in the best region, in the middle of the
server's half, on 43 of 54 serves. The strict rule passes on 4 of 6 runs: seed 5
fails in both sports. In pickleball its greedy cell misses a region of 2 to 5
cells on 3 serves, by at most 0.38 points. In tennis the policy spreads its
probability over the tied cells of the plateau, so its most likely cell is
fragile: it misses the region on 8 of 9 serves, by up to 6.6 points, while the
expected regret of the same run stays under 0.8 points. With a 1-point
tolerance on the greedy regret, 5 of 6 runs pass. The agent learns the answer of
Section IV-H from the reward alone: the best return is the middle, not the lines.

## What It Does Not Show Yet

**A weak test in tennis.** The tennis region holds 40 to 45 of 150 cells, so a
tennis pass is weak evidence; pickleball, with 2 to 8 cells, is the sharper test.

**Drift.** On a reward that is almost always 1, PPO drifts after about 100k
steps. The fixed run length avoids the drift but does not explain it.

**How much of the policy is in the region.** The greedy cell measures the region
with a single point. The probability that the policy puts inside the best region
would measure it directly, and needs a new run: it is a follow-up.

**Scope.** The environment covers the deuce side, one shot, and a server who
neither moves nor adapts. The ad side, rallies, self-play, and pruning the grid
with the results of Section IV-H, the third phase of the project, are future
work (Section X).

Details: [Gdoc, Research results tab, Experiment 6](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0#heading=h.kntokyu5w697).

# Discussion and Limitations

**Neglected aerodynamics.** Measured drag and lift coefficients for tennis
balls show that aerodynamic forces are not negligible relative to gravity,
particularly at high speed and with topspin [@cross2014drag]. Because the
no-drag model of Section III-C systematically shortens actual flight
distance relative to the no-drag prediction for a given launch speed and
angle, the trajectory-feasibility and flight-time estimates in this paper
should be read as first-order approximations; the direction of the bias is
known, but we have not quantified its magnitude for the specific speed and
spin ranges of interest here.

**Point-mass players and fixed speed.** Modeling players as points with a
single maximum speed $v_p$ ignores acceleration and deceleration dynamics,
the direction-dependence of movement speed (lateral versus forward
recovery), and anticipation effects, whereby experienced players begin
moving before ball contact based on early postural cues, which would
effectively reduce $t_r$ below the simple-reaction-time baseline used
here.

**Error model coupling.** The stroke-error parameters $\sigma_\theta,
\sigma_v, \sigma_\phi$ are treated in Section III-C as fixed, exogenous
inputs, but in practice they likely increase with target difficulty, for
example, wider angles or greater required pace probably carry larger
execution error than safe, central targets. The current framework does
not model this coupling; doing so would require calibrating
$\sigma(\cdot)$ as a function of target geometry from empirical stroke
data, which is a natural extension. The 2D model has the same limit: its
spread $(\sigma_x, \sigma_y)$ does not depend on how far the returner runs
before hitting, a coupling that would make the serve a strategic choice.

**Straight-line flight and no bounce.** The 2D model of Section IV-H flies the
ball in a straight line at a constant ground speed, and asks the server to
reach the landing point by the time the ball lands. Real returns slow down and
bounce, and the server can play the ball after the bounce, which gives them
more time. The model therefore understates the server's reach, and with it the
value of aiming away from the server: the headline result holds as long as
that reach stays small compared with the spread of the return.

**Saturation.** At the default settings many aim points reach a
$P_{\mathrm{win}}$ close to 1 in tennis, so the highest value cannot rank
settings (Section VII-A), the best region is large ([@fig-winner-map]), and a
learned policy can pass in tennis without being precise (Section VIII). A
faster server, a slower ball, or a larger spread sharpens the comparison;
whether the defaults are realistic or need calibrating is open.

**Deterministic reachability.** Section IV-C treats $R(c)$ as a strict
indicator. In practice a player's reaction time and top speed are
themselves variable, so a graded, probabilistic reachability model, for
example, a smooth function of the margin $T_f(c) - t_r - d(O,c)/v_p$
rather than a step function, would likely produce a more realistic
composite score near the boundary of the reach envelope.

**Grid resolution versus variance.** A finer grid better approximates the
continuous placement decision but requires more Monte Carlo samples per
cell to control estimation variance in $P_{\mathrm{in}}$, and the
resulting larger payoff matrix increases the cost of the game-theoretic
solution step in Section V. The 2D model uses common random numbers and a
fixed grid ($10 \times 15$ cells for tennis and $9 \times 10$ for pickleball,
Section IV-H), and we do not study this trade-off quantitatively here.

# Conclusion and Future Work

We proposed a reduced-order framework for optimal shot placement in tennis
and pickleball and computed its two-dimensional instance for the return of
serve. The winner probability of every aim point is estimated by Monte Carlo
as one joint event, on common random numbers, and the best placement is a
region of statistically tied aim points. At the default settings the best
return is the middle of the server's half in both sports, not a spot near the
lines: landing in matters more than landing far from the opponent. The error
spread matters most; the ball and player speeds matter mainly with a slow ball
and a fast server, and with no reaction time they act only through their ratio
(Proposition 1). A one-step reinforcement learning environment built on the
same model reproduces the winner probability map from the reward alone, and a
PPO agent trained on it aims at the same middle region. The general framework,
a trajectory model, a grid score with reachability, and a zero-sum game
connected to empirical evidence of minimax play in professional tennis
[@walker2001minimax], frames how this model extends to full rallies, and the
same parametrized model covers both sports.

The results are computational: the parameters come from literature ranges and
round values, not from tracking data, so they show how the model behaves rather
than how players play. The most immediate next steps are: (i) making the spread
of a return grow with the ball speed and with the distance the returner runs;
(ii) calibrating the spread $(\sigma_x, \sigma_y)$ and the reachability
parameters $(t_r, v_p)$ from tracking data or from experts, rather than from
literature ranges; (iii) replacing the deterministic reachability with a graded,
probabilistic version; (iv) extending the reinforcement learning environment to
the ad side, to rallies in which both players move, and to self-play, which
leads to the post-shot positioning game of Section V-C; (v) explaining the
simulated maps in closed form, for example by symbolic regression; and (vi)
extending the geometric model to doubles court dimensions and validating the
minimax-equilibrium hypothesis of Section V-B directly on placement data,
rather than only on serve-direction data.

# Appendix A: Proof Sketch of Proposition 1 {.unnumbered}

Take the model of Section IV-H. The landing points $(X_k, Y_k)$ are drawn from the
target and the spread, so their distribution does not depend on $v_b$ or $v_p$,
and whether a sample lands in the server's half $H$ depends only on $(X_k, Y_k)$.
With $d_k = \lVert (X_k, Y_k) - (x_1, y_1) \rVert$ the flight distance and
$r = v_p / v_b$, the reach rule of Section IV-H reads

$$
\begin{aligned}
\lVert (X_k, Y_k) - P \rVert
&\le v_p \max\left(0, \, \frac{d_k}{v_b} - t_r\right) \\
&= \max(0, \, r \, d_k - v_p t_r) .
\end{aligned}
$$

So each event $W_k$ depends on the two speeds only through $r$ and the reaction
distance $v_p t_r$. With $t_r = 0$ the reaction distance is 0 and only $r$ is left.
$P_{\mathrm{win}}$ is the probability of this event, so it depends on $v_p$ and
$v_b$ only through $r$, and so do the best region and the best target, which are
functions of the map. With $t_r > 0$ the map depends on both $r$ and $v_p t_r$.

# Appendix B: Experiment Index {.unnumbered}

The paper holds the model and the reasoning; the runs, the figures, the settings
strings, and the decisions behind each result are in the project's research log,
a Google Doc whose Research results tab has one entry per experiment:

<https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0>

Notebook $NN$ is `notebooks/racket_sports_NN_*.ipynb` of the accompanying code.


: Where the details of each result are kept

| Section | Result | Notebook | Research log |
| :--- | :--- | :--- | :--- |
| IV-G | In-bounds probability | 01, Part 1 | [Experiment 1](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0) |
| IV-H | Winner probability map | 01, Part 2 | [Experiment 2](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0) |
| VII-A | Sweeps and serves | 05 | [Experiment 3](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0) |
| VII-A | Speed ratio, reaction time | 05, Part 3 | [Experiment 4](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0#heading=h.xvdn4n7zo860) |
| VIII | Reinforcement learning | 06 | [Experiment 6](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0#heading=h.kntokyu5w697) |
