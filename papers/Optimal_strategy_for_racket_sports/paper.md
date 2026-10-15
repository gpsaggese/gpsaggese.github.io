---
title: "Optimal Shot Placement in Racket Sports: A Game-Theoretic Framework"
author:
  - name: "`Pranav Shashidhara, Giacinto Paolo (GP) Saggese, Aahan Rembersu`{=typst}"
    department: "Department of Computer Science"
    organization: "University of Maryland, College Park"
    location: "College Park, MD, USA"
    email: "`pshashid@umd.edu, gsaggese@umd.edu, arembers@terpmail.umd.edu`{=typst}"
abstract: |
  Choosing where to place a shot is a central tactical decision in tennis and
  pickleball, yet it is rarely studied as a single, tractable optimization
  problem. Existing approaches either simulate full three-dimensional ball flight
  with Monte Carlo sampling, which is computationally expensive and hard to
  reason about analytically, or mine historical tracking data to describe what
  players already do, without prescribing what they should do.
  <!-- -->
  This paper proposes a reduced-order framework that models shot trajectories in
  a single vertical plane, discretizes the court into a grid of candidate target
  cells, and scores every cell by combining a Monte Carlo estimate of the
  probability that the ball lands in bounds with a deterministic estimate of
  whether the opponent can reach it in time. We further cast repeated placement
  decisions as a zero-sum game between striker and returner, connecting the
  framework to prior empirical evidence of minimax play in professional tennis.
  <!-- -->
  Because tennis and pickleball share the same geometric and kinematic structure
  and differ, in our model, essentially only in maximum ball speed and court
  scale, a single parametrized framework covers both sports. We illustrate the
  scoring mechanism with a small worked example and discuss the limitations of
  the 1D/2D approximation relative to full 3D simulation.
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

The framework rests on three simplifications.

- Because the source point, the aim point, and the net lie approximately on
  a single line for a given aim direction, ball flight can be modeled as
  one-dimensional motion within the vertical plane that contains that line,
  rather than as full three-dimensional flight.
- The court is discretized into a grid of candidate target cells, and the
  probability that a given cell is a legal landing spot is estimated by Monte
  Carlo sampling over a low-dimensional error model (angle and speed of the
  stroke), rather than by an analytically intractable closed-form
  distribution.
- Whether the opponent can reach a given cell in time is treated as a
  deterministic function of travel distance, reaction time, and maximum
  movement speed, rather than as a further source of randomness.

The main contributions of this paper are:

- A 1D/2D reduction of the trajectory-feasibility problem that maps a
  desired target and net-clearance constraint to a feasible set of launch
  angles and speeds, together with an error-propagation model that turns
  execution noise into a landing-location distribution (Section III).
- A grid-based scoring framework that combines a Monte Carlo estimate of
  in-bounds probability with a deterministic reachability indicator into a
  single composite score per target cell, applicable to both serving and
  rallying (Section IV).
- A game-theoretic extension that frames repeated placement decisions as a
  zero-sum game and connects the resulting mixed-strategy intuition to
  existing empirical evidence of minimax play at the professional level
  [@walker2001minimax] (Section V).
- A demonstration that tennis and pickleball can be described by the same
  parametrized model, differing primarily in maximum ball speed and court
  scale, illustrated with a small worked example that exposes qualitatively
  different optimal regimes for the two sports (Sections VI and VII).

The paper is organized as follows. Section II reviews related work. Section
III formalizes the problem. Section IV describes the grid-based scoring
methodology. Section V develops the game-theoretic layer. Section VI works
a small illustrative example by hand. Section VII discusses the tennis and
pickleball parametrizations. Section VIII discusses limitations, and
Section IX concludes.

# Related Work

**Ball trajectory physics.** The aerodynamics of racket-sport balls, including
drag and Magnus-effect lift due to spin, have been measured directly using
high-speed cameras and wind-tunnel experiments [@cross2014drag]. These
measurements show that drag and lift forces are not negligible relative to
gravity, particularly at the higher speeds reached by serves and hard
groundstrokes. Our framework deliberately ignores these effects to keep the
trajectory problem analytically tractable, at the cost of introducing a
systematic bias that we return to in Section VIII.

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
  This is the most consequential simplification; Section VIII quantifies
  its likely impact using measured drag and lift coefficients
  [@cross2014drag].
- **Bounded reaction time.** A player requires a fixed latency after ball contact
  before beginning to move. We use the 190-250 ms range reported for simple
  visual reaction time as a plausible baseline [@woods2015reaction], noting that
  this is can be conservative estimate: anticipation and cue reading can shorten
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
P_{\mathrm{in}}(c) \approx \frac{1}{K} \sum_{k=1}^{K} \mathbb{1}[\,
\mathrm{land}(\theta_k, v_{0,k}, \phi_k) \in c \,] ,
$$

where a sample counts only if the simulated landing point falls within
both cell $c$ and the legal court boundary.

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

where $T_f(c)$ is the ball's flight time to $c$ from Section III-B. Unlike
$P_{\mathrm{in}}$, $R$ is not estimated by sampling: given fixed $t_r$,
$v_p$, and $T_f(c)$, whether the opponent's minimum possible arrival time
is within the flight-time budget is a yes/no geometric fact, not a random
outcome. Randomness in the opponent's true reaction time or top speed can
be incorporated by treating $R(c)$ as a probability rather than an
indicator; we retain the deterministic form here for tractability and note
the extension in Section VIII.

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
the trajectory model. A simpler model, which we use as a first building block
of the computational implementation, places the uncertainty directly on the
landing point: a shot aimed at $(x_2, y_2)$ lands at $(X, Y)$ with

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
sports, in `notebooks/racket_shot_probability_utils.py` of the accompanying
code.

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
future work (Section IX).

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
through a small, hand-computed example. It is intended purely to
illustrate the mechanism, not as a simulated experiment; a full
computational implementation of the grid Monte Carlo pipeline is future
work (Section IX).

## Setup

Figure 1 shows the trajectory-feasibility geometry underlying the example:
a contact point $S$ at height $h_0 = 1.0$ m, a net at $d_{\mathrm{net}} =
12$ m with $h_{\mathrm{net}} = 0.914$ m, and a target $T$ at $d_T = 20$ m.
For a launch angle $\theta = 8^\circ$ and speed $v_0 = 22.9$ m/s, the
resulting trajectory clears the net by about $0.40$ m and lands at $T$
after a flight time $T_f = 0.88$ s; the dashed curves show four perturbed
trajectories under $\pm 1.5^\circ$ and $\pm 3^\circ$ angle error combined
with $\mp 5$-$8\%$ speed error, and the shaded segment shows the resulting
spread of landing points.

![1D trajectory-feasibility geometry: contact point $S$, net, target $T$, the nominal trajectory (solid), four perturbed trajectories (dashed), and the resulting landing spread.](figures/trajectory_schematic.png)

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
Figure 1), the available movement time after reacting is $T_f - t_r = 0.6$
s, giving a reach radius $v_p (T_f - t_r) = 0.9$ m. Cells $c_1$ and $c_2$
fall within this radius ($R = 1$); cells $c_3$-$c_5$ do not ($R = 0$).
Table II reports $P_{\mathrm{in}}$, $R$, and the composite score $S$ for
each cell; Figure 2 (left panel) shows the same values graphically. The
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
P_{\mathrm{in}}(c)$ everywhere (Table II, Figure 2, right panel). The
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

![Toy-example scores for five candidate target cells, tennis (left) versus a fast pickleball net exchange (right). Gray bars: in-bounds probability $P_{\mathrm{in}}$; colored bars: composite score $S$; star: argmax cell.](figures/toy_example_scores.png)

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
"sport" as a parameter setting rather than a structural choice in future
computational implementations of the framework.

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
data, which is a natural extension.

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
solution step in Section V. We do not study this trade-off quantitatively
here; doing so requires the computational implementation left to future
work.

# Conclusion and Future Work

We proposed a reduced-order framework for optimal shot placement in tennis
and pickleball that avoids full 3D physical simulation by modeling
trajectories in a single vertical plane, scores candidate targets on a
discretized court grid by combining Monte Carlo in-bounds probability with
deterministic opponent reachability, and extends the resulting scoring
rule to a game-theoretic setting connected to existing empirical evidence
of minimax play in professional tennis [@walker2001minimax]. We showed
that the same parametrized model, differing only in maximum ball speed and
court geometry, produces qualitatively different optimal placement regimes
for tennis and pickleball in a small worked example.

The framework as presented is a methodology rather than a validated
system: no computational implementation or empirical evaluation has yet
been carried out. The most immediate next steps are: (i) implementing the
grid Monte Carlo pipeline computationally to move from the hand-worked
example of Section VI to results at realistic grid resolution; (ii)
calibrating the error model $(\sigma_\theta, \sigma_v, \sigma_\phi)$ and
the reachability parameters
$(t_r, v_p)$ from real trajectory-tracking data rather than from
literature ranges; (iii) replacing the deterministic reachability
indicator with a graded, probabilistic version; (iv) solving the post-shot
positioning game of Section V-C as a repeated, multi-shot stochastic game
rather than a single-shot approximation; and (v) extending the geometric
model to doubles court dimensions and to a validation of the
minimax-equilibrium hypothesis of Section V-B directly on placement data,
rather than only on serve-direction data.
