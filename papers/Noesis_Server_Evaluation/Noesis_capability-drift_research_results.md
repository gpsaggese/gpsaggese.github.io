# Abstract

Managed inference exposes a model through a stable API, but the service
behind that endpoint can change. This paper evaluates a lightweight
Noesis monitor for detecting two kinds of change: loss of
contract-relevant capability and shifts in response behavior. Tier 0
uses a frozen set of exactly graded canaries with a fixed-window Fisher
test; a preliminary Bernoulli CUSUM reads the same pass/fail stream for
earlier warning. Tier 1 compares repeated stochastic outputs with
Hamming-kernel MMD. In controlled trials, the canary detector found
large planted degradations and full substitutions without alerting on
healthy controls. In live Llama experiments, the canaries separated 70B
from 8B performance, while the fingerprint distinguished 70B and 8B
response pools and remained quiet across same-model comparisons. CUSUM
detected planted declines earlier than Fisher in the pilot, although it
was calibrated with stronger baseline knowledge. The results support
inexpensive continuous canaries with lower-frequency distributional
audits. They do not establish model identity or long-run production
false-alarm rates; those require broader benchmarks and multi-day
collections.

# 1. Introduction

Model inference is commonly purchased through an API that names a model
and, in some cases, a provider. That name is a contractual promise, not
a direct observation of the system that produced an answer. A buyer
cannot inspect the model weights, numerical precision, or serving path
behind each request. The delivered service may therefore change after a
model release, an infrastructure update, a routing decision, or a
substitution. Prior work has documented task-level behavior changes
under stable product names (Chen, Zaharia, and Zou, 2023), tested
whether API outputs match a reference model (Gao, Liang, and Guestrin,
2025), and analyzed incentives to serve a cheaper model under a more
valuable identity (Cai et al., 2025). These studies motivate continuous
measurement, but no single signal determines whether a particular
inference contract was fulfilled.

Noesis treats capability, latency, and reliability as measurable
contract terms rather than incidental telemetry. This creates a
practical monitoring problem. A useful detector must identify a material
loss soon enough to inform routing or settlement, avoid frequent alerts
under ordinary model variation, preserve evidence for later review, and
cost much less than the workload it protects. It must also distinguish
two claims that are related but not equivalent. A decline in graded task
performance is evidence of capability loss. A change in the distribution
of responses is evidence of behavioral drift and may indicate a
different model or configuration. Behavioral change does not by itself
prove lower quality, substitution, or provider intent.

We study a three-tier monitor designed around that distinction. Tier 0
repeatedly runs a small, frozen panel of contract-specific questions
with exact grading. The fixed-window Fisher detector compares each block
of 60 grades with a frozen healthy baseline. A preliminary CUSUM
detector analyzes the same pass/fail stream one result at a time and
serves as a possible earlier-warning companion. Tier 1 examines repeated
stochastic outputs with MMD and a Hamming kernel, a statistical
comparison of whether two response patterns appear to come from the same
distribution. It runs after a persistent Tier 0 anomaly or on a
lower-frequency schedule. Tier 2 reserves broad benchmark and agentic
suites for onboarding, model releases, and incident review. The
experiments in this paper evaluate the first two tiers using controlled
interventions and small live-provider collections. Tier 2 remains
proposed future work.

## 1.1 Research question

Can a router-embedded monitor detect material capability loss or
behavioral change with a small, reproducible probe budget while keeping
healthy-control alerts low enough for operational use?

## 1.2 Contributions

This paper makes four contributions. First, it defines contract
under-delivery, capability regression, behavioral drift, and probable
substitution as distinct evidence classes. Second, it specifies a
three-tier monitoring architecture that combines a five-family
known-answer panel with a lower-frequency behavioral fingerprint. Third,
it evaluates a fixed-window Fisher detector, a preliminary one-sided
Bernoulli CUSUM detector, and an MMD test with a Hamming kernel under
controlled degradation and substitution scenarios. Finally, it reports a
paid 240-call warm-up and two live fingerprint collections while
preserving provider, model, cost, latency, and response evidence.
Throughout the analysis, distributional change is treated as a signal
for investigation rather than proof of lower quality or malicious
conduct.

## 1.3 Scope

The evaluation focuses on delivered capability and behavioral change.
Availability, latency, and cost are retained as context, but we do not
attempt a complete service-level assessment or model attestation.
Results apply to the tested routes, prompts, and collection windows.
Broad Tier 2 benchmarks and multi-day operation are outside the current
study.

# 2. Background and Problem Definition

## 2.1 Inference as a service contract

Noesis treats an inference request as part of a service contract rather
than as an isolated text-generation call. The contract may name a model
and provider, but it can also specify observable requirements for
capability, availability, latency, and cost. A buyer therefore needs
evidence about the service actually delivered, not only the endpoint
label shown in an API request.

Monitoring is organized around a fixed series identity: provider, model
identifier, probe-bank version, grader version, and runtime
configuration. Comparisons are valid only within the same identity. A
change to the prompt bank, grader, decoding parameters, or named model
starts a new series or requires an explicit re-baseline. This rule
prevents a configuration change from being mistaken for model drift.

Each monitored request produces an immutable observation containing the
prompt identity, returned response, requested and served route,
timestamp, status, latency, token use, and cost. When a question has a
known answer, the observation also receives a deterministic grade:
correct, incorrect, or ungradable. The monitor can evaluate these
outputs and metadata, but it cannot directly inspect the model weights
or the complete inference stack. Its conclusions must therefore concern
observable service behavior.

## 2.2 Evidence classes

We use four evidence classes to keep operational findings separate from
stronger claims about model identity or provider intent.

**Contract under-delivery.** An observed service fails one or more terms
of the inference contract during an evaluation window. The failure may
concern capability, availability, latency, route attribution, or cost.
This is the broadest category and does not identify the underlying
cause.

**Capability regression.** Accuracy on a frozen set of contract-relevant
tasks falls materially below a healthy baseline. In this study, the
evidence comes from exactly graded canaries and must satisfy both a
statistical threshold and a minimum practical decrease. The conclusion
applies to the tested task distribution, not to every possible use of
the model.

**Behavioral drift.** The distribution of responses changes relative to
a reference collection, even when no answer key is available. Behavioral
drift has no built-in direction: a changed model may be better, worse,
or simply different. It is therefore an identity or configuration signal
rather than a quality verdict.

**Probable substitution.** Capability and behavioral evidence are
jointly consistent with requests being served by a known alternative
model or route under controlled conditions. We use the term probable
because text-only evidence cannot establish the exact weights used for
an individual request. The same pattern could also arise from
quantization, fine-tuning, decoding changes, routing changes, or another
implementation difference.

These categories are deliberately not collapsed into a single score. A
model can preserve canary accuracy while changing its response
distribution, or lose accuracy without resembling the weaker reference
model used in an experiment. Agreement between the two evidence lanes
strengthens the case for investigation, while disagreement remains
useful diagnostic information.

## 2.3 Black-box monitoring and related work

Longitudinal evaluation is necessary because a stable product name does
not imply stable behavior. Chen, Zaharia, and Zou (2023) measured
different versions of GPT-3.5 and GPT-4 on a common task set and found
substantial changes across tasks over time. Their results motivate
repeated evaluation, but a benchmark score alone cannot determine
whether an inference contract was fulfilled or explain why performance
changed.

Gao, Liang, and Guestrin (2024) formalized Model Equality Testing as a
two-sample problem: an auditor collects outputs from a reference
distribution and from an API, then tests whether the two collections
appear to come from the same distribution. Their results show that
maximum mean discrepancy with a simple Hamming-based string kernel can
detect several forms of model or serving-stack distortion with
relatively few samples. This method supplies the basis for the
behavioral-fingerprint lane in Noesis. It can show that two response
distributions differ, but it does not determine whether the difference
reduced task quality.

Cai et al. (2025) studied model-substitution auditing under black-box
and adversarial conditions. They compared text-output tests, benchmark
evaluation, log-probability analysis, and hardware-assisted
verification. Their analysis highlights an important limit for Noesis:
text-only methods are widely applicable but can lose sensitivity under
subtle, randomized, or adaptive substitutions. Log probabilities and
trusted hardware can provide stronger identity evidence when available,
but they require provider support and different trust assumptions.

Noesis addresses a narrower operational gap. It combines
contract-specific capability checks with a separate distributional test,
preserves the evidence needed to review each alert, and places both
methods inside a cost-aware monitoring schedule. The purpose is not to
replace attestation. It is to determine whether delivered service
remains consistent with an agreed baseline and whether a deeper audit
should be triggered.

## 2.4 Threat model and non-goals

We consider an inference endpoint that may change because of routine
releases, infrastructure updates, quantization, fine-tuning,
load-dependent routing, full substitution, or partial substitution.
These changes may be announced or undisclosed, and they may be benign,
economically motivated, or accidental. The monitor must also tolerate
ordinary stochastic variation, transient failures, and incomplete
responses without treating each event as degradation.

The initial threat model does not assume a fully adaptive provider that
can always recognize and selectively answer audit prompts. The canary
bank is versioned and scheduled privately to reduce simple gaming, but a
fixed bank may eventually become recognizable. Bank rotation, buyer-like
probe formats, and stronger attestation remain future defenses.

The monitor does not identify exact model weights, infer provider
intent, or provide cryptographic proof of execution. It also does not
certify performance outside the selected probe families. An alert means
that the observed evidence crossed a frozen decision rule and should
trigger review or deeper testing. It is not, by itself, proof of fraud
or a complete service-level verdict.

# 3. Monitoring Architecture

Noesis uses a layered monitor because no single test can answer every
quality-of-service question at acceptable cost. The architecture
separates evidence collection from statistical evaluation and separates
capability evidence from behavioral evidence. Cheap, interpretable
checks run frequently. More expensive distributional or benchmark-based
tests run on a schedule or after an earlier signal persists. Figure 1
summarizes this flow.

<img
src="Noesis_capability-drift_research_results.figs/image2.png"
style="width:6.25in;height:3.47222in" />

*Figure 1. Noesis monitoring architecture. A frozen collection plan
produces immutable observations that can support capability and
behavioral tests. Tier 2 is reserved for deeper audits. Alerts preserve
evidence for review and do not automatically penalize a provider.*

## 3.1 Design principles

The architecture follows four principles. First, the collection plan is
frozen before execution. The plan identifies the provider, model,
prompt-bank version, grader, decoding settings, call budget, and stop
conditions. A material change creates a new campaign identity rather
than silently modifying an active run.

Second, collection is separate from evaluation. Provider responses and
metadata are stored before a detector is run, allowing the same
preserved evidence to be regraded or reanalyzed without making another
paid call. When a prompt and decoding configuration support both
analyses, one response can contribute to both the capability and
behavioral lanes.

Third, the two evidence lanes remain independent. Capability tests ask
whether the endpoint still succeeds on selected tasks. Behavioral tests
ask whether the response distribution changed. Their outputs are
interpreted together, but they are not averaged into a single score
because they support different claims.

Finally, monitoring remains outside the market-decision path during this
study. It does not create orders, change contract verdicts, update
provider reputation, or automatically reroute traffic. This boundary
allows experimental thresholds to be evaluated without creating an
economic penalty.

## 3.2 Evidence collection and persistence

A monitoring run begins with a versioned target and probe schedule. The
target fixes the provider and model route, while the schedule selects
prompts from the approved bank under a bounded call budget. Live
execution requires explicit authorization. Route attribution, token
limits, cost limits, timeout limits, and provider-error limits are
checked during collection.

Each call becomes an immutable observation containing the requested and
served route, prompt identity, raw completion, deterministic grade when
available, timestamp, latency, token counts, billed cost, and status.
The observation is durably written before the next call begins. This
ordering protects the audit trail if a run is interrupted and makes
resume behavior explicit rather than reconstructing missing evidence
from memory.

Observations are aggregated into non-overlapping monitoring windows.
Each window records graded accuracy, availability, latency percentiles,
token use, cost per attempted task, and cost per correct task. Windows
with too little gradable evidence are marked insufficient instead of
being interpreted as capability degradation. Baselines are scoped to the
complete series identity so that different models, banks, graders, or
runtime configurations are not compared as if they were interchangeable.

## 3.3 Tier 0: contract canaries

Tier 0 is the continuous, low-cost layer. It repeatedly samples a small
frozen panel of questions whose answers can be graded exactly. The panel
spans five capability families so that the aggregate result is not
determined by one narrow task type. Because grading is deterministic,
every pass or failure can be traced to a specific prompt and answer
rule.

The fixed-window Fisher detector is the primary interpretable decision
rule. It compares the number of correct and incorrect grades in a
current window with a frozen healthy baseline. A window is flagged only
when the decrease is both statistically unlikely under the healthy
baseline and large enough to matter in practice. The first flagged
window produces a warning, and a second consecutive flagged window
confirms the experimental NERFED state. Exact sample sizes and
thresholds are specified in Section 4.

CUSUM is evaluated as a sequential companion to Fisher. It consumes the
same pass/fail observations but updates after every grade rather than
waiting for a complete window. This can reduce detection delay, and it
requires no additional provider calls. However, its alarm threshold must
be calibrated against healthy streams, and the current CUSUM result is
preliminary. The architecture therefore treats Fisher as the main
fixed-window detector and CUSUM as an early-warning comparator rather
than a proven replacement.

## 3.4 Tier 1: behavioral fingerprints

Tier 1 tests for broader response changes when an answer key is
unavailable or when Tier 0 produces a persistent anomaly. It may also
run on a lower-frequency schedule to detect changes that preserve canary
accuracy. The monitor collects repeated stochastic responses for a
frozen prompt set and compares a current pool with an independent
reference pool.

The comparison uses maximum mean discrepancy with a Hamming kernel. The
kernel measures similarity between response strings, while MMD
summarizes whether within-pool responses are more similar than responses
drawn across the two pools. A permutation test estimates how often a
difference at least this large would appear if the pool labels were
exchangeable. The output is evidence of a distributional difference, not
a declaration that one pool is better.

Tier 1 does not require a deterministic grader, which makes it useful
for open-ended behavior. It is also more expensive than a small canary
panel because each prompt requires repeated samples and the offline
permutation analysis is computationally heavier. For that reason, it is
scheduled less frequently and serves as a confirmation or investigation
layer.

## 3.5 Tier 2: broad evaluation

Tier 2 consists of broad benchmark suites, agentic tasks, or human
review. These evaluations cover more behaviors than the frozen canaries
and can help characterize an incident, but they are slower, more
expensive, and harder to grade reproducibly. The proposed portfolio runs
Tier 2 during provider onboarding, major model releases, and incident
review rather than on every monitoring cycle.

Tier 2 is part of the architecture but not part of the completed
experimental evidence in this paper. The whitepaper therefore makes no
claim about its detection power, cost, or operational false-alarm rate.
A future evaluation should define the exact benchmark suite,
task-specific graders, execution budget, and escalation criteria before
Tier 2 affects a service decision.

## 3.6 Alert interpretation and operational response

The detector outputs are interpreted as an evidence portfolio. A
capability alert without behavioral drift supports a task-performance
regression claim. Behavioral drift without a capability alert supports a
configuration or identity-change claim. When both signals align with a
controlled alternative model, the result supports probable substitution
and a deeper audit. Availability, latency, and cost remain separate
supporting dimensions rather than being folded into the model-quality
verdict.

Every alert retains the frozen plan, raw observations, aggregate
metrics, detector version, thresholds, and decision trace. This record
allows an independent reviewer to reproduce the decision and determine
whether a configuration change, provider incident, or data-quality
problem offers a better explanation.

The operational response is intentionally conservative. A warning may
increase sampling or trigger Tier 1. A confirmed capability alert or
strong behavioral change may trigger Tier 2, manual review, or a
temporary routing recommendation. Automatic settlement, provider
penalties, and reputation changes remain outside the evaluated system.

# 4. Experimental Methodology

We organize the work by phase and evidence lane. Phase 0 validates
detectors on synthetic evidence. Phase 1 collects live, exactly graded
capability data for Lane 1, while Phase 2 collects live stochastic
responses for Lane 2 behavioral fingerprints. The offline CUSUM
comparison extends Phase 0 by applying a sequential rule to the same
simulated pass/fail streams.

## 4.1 Experimental questions

The experiments address four questions. First, does the capability
detector remain quiet on healthy evidence while detecting material
planted degradation? Second, do the frozen canaries distinguish
capability across live model and provider routes? Third, can a
grader-free fingerprint distinguish a 70B response distribution from an
8B distribution without flagging independent samples from the same 70B
route? Fourth, can a sequential CUSUM rule detect the same planted
capability losses earlier than the fixed-window Fisher rule under a
comparable finite-horizon false-alarm target?

All prompts, routes, sample counts, statistical parameters, budgets, and
stop conditions were fixed before each campaign. Provider calls produced
immutable observations. Grading and statistical analysis were performed
afterward from the saved evidence. This separation allowed the same
evidence to be reanalyzed without additional paid calls and prevented
detector tuning from changing the data-collection process.

## 4.2 Phase 0: synthetic capability validation

Phase 0 tests the capability detector under known conditions before
interpreting live results. Each simulated trial begins with five healthy
baseline windows. Every window contains 60 binary grades, and the
healthy success probability is 0.95. The intervention period contains 15
additional windows, or 900 post-change grades. We ran 1,000 trials per
scenario.

The scenario panel includes a healthy A/A control, a full weak-model
swap with success probability 0.60, a small immaterial decline, a
decline at the decision boundary, gradual degradation, and partial
substitutions affecting 10, 25, or 50 percent of observations. Separate
latency-only and price-only controls verify that non-capability changes
do not enter the quality verdict. Availability failures and ungradable
responses test whether missing evidence is reported as insufficient
instead of being treated as an incorrect answer. Scenario labels,
intervention times, and substituted row identities are retained by the
evaluator and are not exposed to the detector.

For each completed window, the primary detector forms a two-by-two table
of correct and incorrect grades in the frozen baseline and the current
window. A one-sided Fisher exact test asks whether current accuracy is
lower. A window is flagged only when the p-value is below 0.01 and the
observed decrease is at least 10 percentage points. The first flagged
window produces a warning. Two consecutive flagged windows are required
for the experimental NERFED state. A separate power sweep varies the
amount of baseline and current evidence to show how query count affects
sensitivity.

## 4.3 Phase 1: live capability-canary collection

The live capability bank contains 60 unique prompts divided evenly
across five families: code tracing, expert science, instruction
following, mathematics, and multi-step reasoning. Each family
contributes 12 prompts. Answers are evaluated with deterministic,
family-appropriate graders, including exact-text, integer, fraction,
multiple-choice, and structured JSON checks. The bank order, prompt
text, grader identity, and grader configuration are hashed into the
campaign plan.

The controlled warm-up crosses every prompt with four pinned routes:
Cloudflare Llama 3.3 70B, Cloudflare Llama 3.1 8B, Cloudflare Llama 3.2
3B, and CoreWeave Llama 3.3 70B. This produces 240 planned calls, with
60 calls per route and 48 calls per capability family across routes. The
Cloudflare 70B route is the primary reference, the 8B and 3B routes
provide smaller-model comparisons, and the CoreWeave 70B route provides
a same-model, different-provider control.

Before the warm-up, a 15-call attribution smoke test checks route
pinning, served-provider metadata, usage accounting, cost accounting,
and durable persistence. Provider fallback is disabled. An answered
request with missing or unexpected route attribution halts collection
because it cannot support the planned comparison. Each observation
records the requested and served route, prompt and grader identity, raw
completion, grade, latency, token use, billed cost, status, and
timestamp. Observations are flushed to durable storage before the next
call. Interrupted campaigns resume from preserved attempts rather than
replaying completed calls.

## 4.4 Phase 2: Hamming-MMD behavioral fingerprints

The fingerprint experiment uses five open-ended prompts covering code
review, constrained narrative, policy writing, scientific explanation,
and systems analysis. For each prompt, the runner collects 12 stochastic
responses at temperature 1.0, giving 60 responses per pool. The initial
live campaign collects two independent Cloudflare 70B pools, reference-a
and reference-b, and one Cloudflare 8B alternative pool. Calls are
deterministically interleaved so pool identity is not confounded with
collection order.

Responses are converted into fixed-length character sequences using the
first 1,000 characters, Unicode tokenization, and padding. We use the
Hamming-kernel MMD implementation from the Model Equality Testing
codebase at pinned commit fd2ee24d75c9fef87debff8caefa0c04d4a5d374. MMD
compares within-pool similarity with cross-pool similarity. Statistical
significance is estimated with 1,000 label permutations, using alpha
0.05. The healthy A/A comparison uses the two independent 70B pools. The
full-swap comparison uses a 70B reference pool and the 8B alternative
pool.

Partial substitution is evaluated by replacing exact counts within a
60-response reference pool: 6 responses for 10 percent, 15 for 25
percent, and 30 for 50 percent. Replacements are balanced across prompts
and selected with versioned deterministic seeds. Each
partial-substitution level is evaluated over 100 seeded trials. Because
these trials reuse the same saved live pools, they measure conditional
sensitivity to mixtures in this collection, not independent
population-level power.

A compressed stability extension adds three more same-session Cloudflare
70B pools, reference-c through reference-e. Together with the original
two pools, this yields ten pairwise same-model comparisons. Each
comparison uses the same frozen 1,000-permutation test. Raw p-values and
Holm-adjusted p-values are both reported so the collection can be
assessed with familywise error control. This extension measures
same-session stochastic stability, not multi-day drift.

## 4.5 Preliminary CUSUM comparison

CUSUM consumes the same binary grades as Fisher but updates after every
observation. The one-sided Bernoulli rule accumulates log-likelihood
evidence favoring a degraded success probability of 0.85 over a healthy
probability of 0.95 and resets the running sum at zero when the evidence
reverses. An alarm occurs when the cumulative score crosses a threshold.

Using 1,000 healthy streams of 900 observations, we calibrated threshold
8.080612 to a 1 percent finite-horizon false-alarm target. We then
compared CUSUM and Fisher on identical seeded streams in 250 trials per
scenario, recording detection and post-change delay. CUSUM received the
assumed healthy rate of 0.95, whereas Fisher estimated its baseline from
five sampled windows; this asymmetry favors CUSUM and is considered when
interpreting the results.

## 4.6 Execution safeguards and reported outcomes

All live runners default to validation-only mode and require an explicit
execution flag, an approved dollar cap, and an API key supplied through
the process environment. The attribution smoke test has a $0.25 cap, the
controlled warm-up a $10 cap, the initial fingerprint campaign a $2 cap,
and the compressed stability extension a $0.50 cap. Collection stops
before another call if attribution cannot be confirmed, the call or cost
limit would be exceeded, two timeouts occur, two consecutive provider
errors occur, the cumulative error rate exceeds 20 percent, grading
fails, or an observation cannot be persisted.

Offline analyses use saved observations and fixed random seeds. The
compressed Hamming-MMD analysis runs sequentially with a 2 GB memory
cutoff, while the CUSUM pilot runs in one process. Primary reported
outcomes are detection rate, false-alert count, first detection window
or query, empirical p-value, and the amount of evidence consumed. Live
reports also retain availability, latency, tokens, and cost as
supporting quality-of-service measurements. No experiment automatically
changes routing, settlement, or provider reputation.

# 5. Results

The experiments produced three kinds of evidence: controlled detector
behavior, live capability measurements, and live behavioral comparisons.
The results are reported separately because a capability loss and a
distributional change support different conclusions.

## 

## 5.1 Synthetic capability-detector validation

The fixed-window detector remained quiet in all 1,000 healthy A/A
trials. Latency-only and price-only controls also produced no capability
alerts. When responses were unavailable or ungradable, the monitor
reported insufficient evidence rather than converting missing
observations into incorrect answers.

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr>
<th style="text-align: left;"><strong>Scenario</strong></th>
<th style="text-align: left;"><strong>Detection rate</strong></th>
<th style="text-align: left;"><strong>Timing or outcome</strong></th>
</tr>
<tr>
<th style="text-align: left;">Healthy A/A</th>
<th style="text-align: left;">0.0%</th>
<th style="text-align: left;">No false alerts</th>
</tr>
<tr>
<th style="text-align: left;">Full weak-model swap</th>
<th style="text-align: left;">100.0%</th>
<th style="text-align: left;">P90 window 2</th>
</tr>
<tr>
<th style="text-align: left;">Threshold-level drop</th>
<th style="text-align: left;">87.3%</th>
<th style="text-align: left;">P90 window 11</th>
</tr>
<tr>
<th style="text-align: left;">Gradual degradation</th>
<th style="text-align: left;">100.0%</th>
<th style="text-align: left;">P90 window 7</th>
</tr>
<tr>
<th style="text-align: left;">Partial substitution, 10%</th>
<th style="text-align: left;">4.4%</th>
<th style="text-align: left;">P90 window 13</th>
</tr>
<tr>
<th style="text-align: left;">Partial substitution, 25%</th>
<th style="text-align: left;">72.7%</th>
<th style="text-align: left;">P90 window 12</th>
</tr>
<tr>
<th style="text-align: left;">Partial substitution, 50%</th>
<th style="text-align: left;">100.0%</th>
<th style="text-align: left;">P90 window 4</th>
</tr>
<tr>
<th style="text-align: left;">Latency only</th>
<th style="text-align: left;">0.0%</th>
<th style="text-align: left;">No capability alerts</th>
</tr>
<tr>
<th style="text-align: left;">Price only</th>
<th style="text-align: left;">0.0%</th>
<th style="text-align: left;">No capability alerts</th>
</tr>
<tr>
<th style="text-align: left;">Unavailable or ungradable</th>
<th style="text-align: left;">N/A</th>
<th style="text-align: left;">Insufficient evidence</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

<img
src="Noesis_capability-drift_research_results.figs/image6.png"
style="width:5.83333in;height:3.86869in" />

*Figure 2. Synthetic fixed-window capability-detector sensitivity across
1,000 trials per scenario. Large and persistent losses were detected
reliably, while small and partial changes were harder to identify.*

Detection rose with intervention size and persistence. The detector
found every full weak-model swap, every gradual decline, and every 50
percent partial substitution. It detected 87.3 percent of declines at
the decision boundary and 72.7 percent of 25 percent substitutions, but
only 4.4 percent of 10 percent substitutions. Small mixtures often leave
too few changed grades in a 60-grade window to satisfy both the
statistical threshold and the required 10 percentage-point decrease.

## 

## 5.2 Live capability-canary results

The controlled warm-up attempted all 240 planned calls exactly once. Of
these, 237 returned provider status ok. Three Cloudflare 8B requests
timed out, and two otherwise successful Cloudflare responses contained
no completion text and were retained as ungradable failures. Every
returned response with attribution matched its requested provider and
model. Total measured cost was $0.01935.

<table>
<colgroup>
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 20%" />
</colgroup>
<thead>
<tr>
<th style="text-align: left;"><strong>Route</strong></th>
<th style="text-align: left;"><strong>Usable</strong></th>
<th style="text-align: left;"><strong>Strict</strong></th>
<th style="text-align: left;"><strong>Substantive audit</strong></th>
<th style="text-align: left;"><strong>Latency / cost</strong></th>
</tr>
<tr>
<th style="text-align: left;">Cloudflare Llama 3.1 8B</th>
<th style="text-align: left;">57/60</th>
<th style="text-align: left;">16/57 (28.1%)</th>
<th style="text-align: left;">19/57 (33.3%)</th>
<th style="text-align: left;">5.25 s / $0.002076</th>
</tr>
<tr>
<th style="text-align: left;">Cloudflare Llama 3.2 3B</th>
<th style="text-align: left;">59/60</th>
<th style="text-align: left;">12/59 (20.3%)</th>
<th style="text-align: left;">22/59 (37.3%)</th>
<th style="text-align: left;">0.82 s / $0.000981</th>
</tr>
<tr>
<th style="text-align: left;">Cloudflare Llama 3.3 70B</th>
<th style="text-align: left;">59/60</th>
<th style="text-align: left;">22/59 (37.3%)</th>
<th style="text-align: left;">36/59 (61.0%)</th>
<th style="text-align: left;">2.49 s / $0.010814</th>
</tr>
<tr>
<th style="text-align: left;">CoreWeave Llama 3.3 70B</th>
<th style="text-align: left;">60/60</th>
<th style="text-align: left;">23/60 (38.3%)</th>
<th style="text-align: left;">35/60 (58.3%)</th>
<th style="text-align: left;">1.01 s / $0.005478</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

<img
src="Noesis_capability-drift_research_results.figs/image5.png"
style="width:5.83333in;height:3.65819in" />

*Figure 3. Substantive capability accuracy by served route in the
240-call live warm-up. Whiskers show 95 percent Wilson intervals, and
each route contributed one completion per prompt.*

The substantive audit showed the expected broad size signal. Cloudflare
70B scored 36 of 59 gradable items, compared with 19 of 57 for
Cloudflare 8B. Across the 56 items gradable for both routes, the exact
paired McNemar p-value was 0.000855. The same test did not find a
difference between the two 70B provider routes: Cloudflare was correct
on 36 of 59 items and CoreWeave on 35 of 60, with a paired p-value of
0.6875 over 59 mutually gradable items. This supports the use of the
smaller model as a capability-degradation control while showing no
detectable provider effect between the two 70B routes in this sample.

The 3B route scored slightly above the 8B route on the substantive
audit. We treat this as a pilot anomaly rather than a general model
ranking. The bank is small, the confidence intervals are wide, and each
route produced only one stochastic response per item. The run also
exposed an instrumentation issue: the frozen strict grader mixed
correctness with exact-format compliance. The original strict scores
were preserved, and the substantive audit was added as a separate
analysis rather than replacing them.

## 

## 5.3 Behavioral-fingerprint results

The Hamming-MMD test separated the Cloudflare 70B reference pool from
the Cloudflare 8B alternative. With 1,000 permutations, the full-swap
empirical p-value was below 0.001. The independent 70B A/A comparison
was not significant.

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr>
<th style="text-align: left;"><strong>Comparison</strong></th>
<th style="text-align: left;"><strong>Detection rate</strong></th>
<th style="text-align: left;"><strong>Empirical p-value</strong></th>
</tr>
<tr>
<th style="text-align: left;">Healthy 70B A/A</th>
<th style="text-align: left;">0.0%</th>
<th style="text-align: left;">.434</th>
</tr>
<tr>
<th style="text-align: left;">Full 70B-to-8B swap</th>
<th style="text-align: left;">100.0% (1 comparison)</th>
<th style="text-align: left;">&lt;.001</th>
</tr>
<tr>
<th style="text-align: left;">Partial substitution, 10%</th>
<th style="text-align: left;">0.0%</th>
<th style="text-align: left;">.253 to .667</th>
</tr>
<tr>
<th style="text-align: left;">Partial substitution, 25%</th>
<th style="text-align: left;">0.0%</th>
<th style="text-align: left;">.139 to .520</th>
</tr>
<tr>
<th style="text-align: left;">Partial substitution, 50%</th>
<th style="text-align: left;">87.0%</th>
<th style="text-align: left;">.012 to .106</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

<img
src="Noesis_capability-drift_research_results.figs/image4.png"
style="width:5.83333in;height:3.77161in" />

*Figure 4. Detection rate as the substituted-response fraction
increases. Capability rates come from independent synthetic trials;
Hamming-MMD rates reuse fixed live response pools and are not a direct
method ranking.*

The partial-substitution trials show a clear sensitivity boundary in
this collection. Replacing 10 or 25 percent of a 60-response pool never
crossed the 0.05 threshold in 100 seeded trials, while 50 percent
substitution was detected in 87 percent of trials. These trials reuse
the same live response pools, so they measure conditional sensitivity to
mixtures in this dataset rather than independent population-level power.

The stability extension compared five independent same-session 70B pools
in all ten pairs. Raw p-values ranged from 0.323 to 0.804, every
Holm-adjusted p-value was 1.0, and neither the raw nor adjusted analysis
produced an alert. Together with the full-swap result, this shows that
the fingerprint distinguished a large cross-model change without
flagging ordinary same-session variation in the available controls. It
does not yet establish a multi-day production false-alarm rate.

## 

## <img
src="Noesis_capability-drift_research_results.figs/image3.png"
style="width:5.83229in;height:4.14861in" />

Figure 5. Hamming-MMD permutation p-values for all ten pairwise
comparisons among five same-session Cloudflare 70B pools. Every raw
p-value remained above 0.05, and all Holm-adjusted p-values were 1.0.

## 5.4 Preliminary CUSUM comparison

CUSUM and Fisher were evaluated on the same seeded binary streams. The
CUSUM threshold was calibrated to a 1 percent false-alarm target over
900 healthy observations. On an independent healthy evaluation it
alerted in 3 of 250 trials, or 1.2 percent, while Fisher alerted in
none.

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr>
<th style="text-align: left;"><strong>Scenario</strong></th>
<th style="text-align: left;"><strong>CUSUM: rate; median
grades</strong></th>
<th style="text-align: left;"><strong>Fisher: rate; median
grades</strong></th>
</tr>
<tr>
<th style="text-align: left;">Healthy A/A</th>
<th style="text-align: left;">1.2% false alerts; 637</th>
<th style="text-align: left;">0.0%; N/A</th>
</tr>
<tr>
<th style="text-align: left;">Full weak-model swap</th>
<th style="text-align: left;">100.0%; 21</th>
<th style="text-align: left;">100.0%; 120</th>
</tr>
<tr>
<th style="text-align: left;">Threshold-level drop</th>
<th style="text-align: left;">100.0%; 93</th>
<th style="text-align: left;">90.4%; 240</th>
</tr>
<tr>
<th style="text-align: left;">Gradual degradation</th>
<th style="text-align: left;">100.0%; 181</th>
<th style="text-align: left;">100.0%; 300</th>
</tr>
<tr>
<th style="text-align: left;">Partial substitution, 25%</th>
<th style="text-align: left;">100.0%; 118</th>
<th style="text-align: left;">80.0%; 300</th>
</tr>
<tr>
<th style="text-align: left;">Partial substitution, 50%</th>
<th style="text-align: left;">100.0%; 47</th>
<th style="text-align: left;">100.0%; 120</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

<img
src="Noesis_capability-drift_research_results.figs/image1.png"
style="width:5.83333in;height:3.65819in" />

*Figure 6. Median post-change grades required for an alarm in the
preliminary paired CUSUM and Fisher simulation. Medians are computed
among detected trials; Fisher detected 90.4 percent of threshold-level
trials and 80.0 percent of 25 percent substitutions.*

Across the planted degradations, CUSUM alarmed with fewer post-change
grades than Fisher; for a full swap, the median was 21 grades versus
120. This advantage should be interpreted cautiously because CUSUM
received the true healthy success probability while Fisher estimated its
baseline. The pilot shows that sequential monitoring can shorten
detection delay, but it does not establish that this CUSUM configuration
is generally superior.

## 

## 5.5 Combined interpretation

Together, the experiments show that the two lanes answer different
questions. The canaries measured directional capability loss and
separated 70B from 8B performance; Hamming-MMD detected a distributional
difference without an answer key. Both signals activated in the full
70B-to-8B comparison, while neither activated in same-model controls.
Small partial substitutions remained difficult for both methods, and
none of the tests identifies exact weights or provider intent.

# 6. Discussion and Operational Implications

Black-box monitoring can support operational decisions without
establishing model identity. Here it measured whether selected
capabilities persisted and whether response behavior matched a
reference. Alerts should therefore be read as reproducible evidence for
investigation, not as a diagnosis of the hidden cause.

## 

## 6.1 Capability and behavior are separate evidence

The capability and fingerprint lanes should be interpreted as two axes.
A capability alert answers a directional question: performance on the
frozen tasks has declined. A fingerprint alert answers a distributional
question: the observed response pattern has changed. The full 70B-to-8B
comparison activated both signals, which is consistent with a material
substitution. The same-model 70B comparisons activated neither signal.
These results show why the two lanes are useful together, but they do
not make the lanes interchangeable.

Disagreement between the lanes is also meaningful. Capability may
decline without a significant fingerprint change if the affected tasks
are narrow or if the behavioral test lacks power. A fingerprint may
change while capability remains stable after a release, decoding change,
or serving-stack update. In that case, the appropriate conclusion is
behavioral drift, not degradation. Noesis should therefore retain both
outputs and their supporting observations instead of combining them into
a single quality score.

This distinction also limits causal claims. Similarity to an 8B
reference can strengthen a substitution hypothesis, but the same pattern
could result from quantization, fine-tuning, prompt processing, decoding
parameters, or another implementation change. Text evidence can justify
escalation and review. It cannot identify exact model weights or
establish provider intent.

## 

## 6.2 Implications for the monitoring portfolio

Tier 0 should remain the routine monitor: its prompts are small, grading
is deterministic, and Fisher produces an interpretable window-level
decision. CUSUM can run on the same grades to increase sampling or
trigger review sooner, but its higher healthy-alert rate and favorable
calibration do not yet justify an independent service verdict. Tier 1 is
better suited to periodic audits and persistent Tier 0 anomalies because
it needs repeated samples and heavier analysis. Tier 2 should be
reserved for onboarding, releases, and incidents.

## 6.3 Cost, delay, and sensitivity

The live capability warm-up shows that the measurement layer can be
inexpensive relative to ordinary inference workloads. All 240 planned
calls cost about two cents under the routes and prices used in this
campaign. This figure demonstrates feasibility for the tested endpoints,
not a universal monitoring price. Cost will change with the model,
provider, prompt length, output length, and sampling schedule.

The more important constraint is statistical sensitivity. A 60-grade
Fisher window detected large and persistent losses but often missed a 10
percent partial substitution. The live Hamming-MMD test also missed 10
and 25 percent mixtures and became reliable only near a 50 percent
mixture in the available pools. These findings imply that subtle or
intermittent changes cannot be solved by scheduling alone. They require
more observations, stronger task selection, a more sensitive
representation, or supporting evidence such as log probabilities or
trusted execution data when available.

There is no single optimal window size. Smaller windows reduce delay but
contain less evidence and can produce unstable estimates. Larger windows
improve power but increase cost and allow a degraded service to continue
longer before confirmation. The correct setting depends on the contract.
A safety-critical or high-value route may justify a larger probe budget
and a lower delay target, while a low-cost route may accept slower
confirmation. Noesis should expose these choices as policy parameters
rather than embedding one threshold for every market.

The offline computation was manageable in the compressed study. The ten
same-model MMD comparisons stayed well below the 2 GB memory limit, with
peak worker memory near 310 MB. This makes scheduled fingerprinting
technically plausible on the current setup. The collection cost and the
number of independent reference pools remain more important limits than
memory in this pilot.

## 

## 6.4 Limitations and threats to validity

The canary bank contains 60 prompts across five families. This is
broader than a single benchmark category but too small to represent all
uses of a general model. The live results compare a limited set of Llama
routes and should not be generalized to other models, providers,
languages, or task distributions without new baselines. A provider could
also recognize a long-lived fixed bank. Private scheduling, versioned
rotation, and buyer-like prompt formats reduce this risk but do not
remove it.

The grading analysis has an additional limitation. The frozen strict
grader penalized some correct answers for including extra reasoning or
violating the requested format. The later substantive audit separated
answer correctness from exact-format compliance, but it was defined
after raw-response review. Preserving both analyses makes the issue
visible; it does not make the later audit preregistered. Future
campaigns should freeze correctness and format-adherence metrics
separately before collection.

The MMD partial-substitution trials are conditional resampling
experiments. They reuse the same 70B and 8B response pools across seeded
mixtures, so the 100 trials at each mixture level are correlated and do
not constitute 100 independent provider collections. Likewise, the
stability extension covers five pools collected in one session. Its
zero-alert result is useful evidence about same-session stochastic
variation, but it does not measure day-to-day releases, load-dependent
routing, regional differences, or long-run false-alarm frequency.

The CUSUM comparison is also favorable to the sequential detector
because it receives the true healthy probability while Fisher estimates
a baseline from sampled windows. Its 1.2 percent healthy alert rate is
close to the calibrated 1 percent target, but 250 validation trials are
not enough to estimate a rare false-alarm rate precisely. A production
comparison should estimate baseline uncertainty, vary the degradation
target, evaluate alternative thresholds, and report average run length
under repeated healthy operation.

Finally, statistical significance is not equivalent to operational
importance. A small p-value can accompany a harmless behavioral change,
while an important change may remain undetected when evidence is sparse.
Detector output should be combined with effect size, availability,
latency, cost, route attribution, and the contract’s materiality
threshold. This is why the architecture preserves an evidence portfolio
and keeps automated penalties outside the current study.

## 

## 6.5 Priorities for the next evaluation

The highest priority is a longitudinal campaign with independent
reference and evaluation pools collected across multiple days. That
campaign should measure healthy alert frequency, detection delay, and
sensitivity to scheduled full and partial interventions under a frozen
protocol. It should also test whether baselines remain valid across
routine provider variation or need time-aware renewal.

The second priority is to turn the CUSUM pilot into a fair sequential
evaluation. Healthy probabilities should be estimated rather than
supplied, thresholds should be calibrated against the same baseline
uncertainty faced by Fisher, and results should be reported across
several loss magnitudes and monitoring horizons. The goal is to
determine when CUSUM should only increase sampling and when it can
support a confirmed alert.

The canary bank should then be expanded and rotated while preserving
exact grading and family balance. Power analysis can determine which
prompts contribute useful separation and how many observations each
contract requires. A broader Tier 2 audit can evaluate whether the
lightweight signals agree with established knowledge, reasoning, coding,
and agentic benchmarks during onboarding or incident review.

Until those evaluations are complete, Noesis should be used to detect
large changes, preserve reproducible evidence, and trigger follow-up,
not to certify model identity or impose automatic penalties.

# 7. Conclusion

This feasibility study shows that lightweight black-box monitoring can
detect material changes in delivered AI inference. The canary lane
identified every large planted degradation in simulation, produced no
alerts in 1,000 healthy trials, and separated live 70B from 8B
capability. The 240-call live campaign cost about two cents on the
tested routes.

The behavioral lane supplied a complementary signal: Hamming-MMD
distinguished the 70B and 8B response pools while remaining quiet across
ten same-model comparisons. Both lanes were less reliable for small
partial substitutions. A preliminary CUSUM shortened detection delay on
the same pass/fail streams, but its favorable calibration and 1.2
percent healthy alert rate require further evaluation.

These results justify a layered design: continuous canaries for
inexpensive capability checks, lower-frequency fingerprints for
behavioral change, and broad benchmarks or human review when risk or an
incident warrants them. The monitor should preserve separate evidence
for capability, behavior, availability, latency, and cost rather than
compressing them into one score. Multi-day collections, rotated canary
banks, fair sequential calibration, and Tier 2 benchmarks are the next
steps before production service-level decisions.
