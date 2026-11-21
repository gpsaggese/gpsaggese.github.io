# Manual review of completed primary contrasts

The primary collection contains all 2,880 planned continuations. These notes
interpret its three harmful decisions that survive Holm correction across
all 48 primary tests. They do not replace the final stability analysis or
complete-population technical audit. Selection for this illustrative review
used the corrected primary results after primary collection was complete;
it did not alter sampling, stopping, or the independent repeat tasks.

| Original decision | Keep wins | Re-decide wins | Repair gain | Unadjusted 95% interval | Holm p |
| --- | ---: | ---: | ---: | --- | ---: |
| krk-m4-1, trial 0, step 3 | 5/30 | 25/30 | +0.667 | [0.428, 0.799] | 0.0000171 |
| krk-m4-1, trial 0, step 5 | 0/30 | 11/30 | +0.367 | [0.180, 0.545] | 0.01498 |
| krk-m4-1, trial 1, step 1 | 0/30 | 11/30 | +0.367 | [0.180, 0.545] | 0.01498 |

At trial 0, step 3, the original reply plays `Ke2` and claims Black can only
reply `Kg1`. The recorded defender instead plays the legal move `Kg2`.
An independent board check finds three legal replies: `Kh2`, `Kg2`, and
`Kg1`. Moving the white king to e2 blocks its own rook on d2, so the rook
no longer covers g2 and h2 as the reply assumes.
The position remains theoretically winnable under the clock-aware oracle,
but keeping this complete reply produces substantially fewer wins under the
tested continuation policy. This demonstrates why the oracle value alone
cannot stand in for a policy-specific repair gain. The intervention changes
the whole reply; these data do not separately identify the effects of its
explanation and its move.

At trial 0, step 5, the original reply plays `Rd1+`. The halfmove clock moves
from 93 before the step to 95 after Black's `Kg2`. The predeclared diagnostic
changes from clock-aware winnable to not winnable. No keep-arm continuation
wins; re-deciding wins 11/30. This is a natural recorded decision, not a
planted blunder.

At trial 1, step 1, the original reply plays `Rh3` and states that the rook
is out of reach. Black's king on g2 immediately captures it with `Kxh3`.
The resulting king-versus-king game ends in insufficient material. Fresh
re-decisions win 11/30, compared with 0/30 when keeping that reply. This is
also a natural failure, not a constructed positive control.

The remaining 45 primary decisions are unresolved after multiplicity
correction; that does not mean their true effects are zero. In particular,
trial 0 step 1 has a helpful point estimate of -0.333 in the primary study,
consistent in direction and size with -0.320 in the earlier independent
validation stage. The primary test does not survive the larger 48-test Holm
family, so it is not reported as a corrected primary discovery.

The two independent full-profile repeats were selected before these outcomes:
trial-0 `krk-m3-1` and `krk-m6-1`. They do not independently repeat the large
`krk-m4-1` effects above. Overall sparse/diffuse summaries are exploratory;
these seven selected failed runs from four starting positions do not support
population-wide failure categories or a validated surrogate.
