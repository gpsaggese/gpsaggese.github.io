# Model-quality Monitoring Research Backlog

- This document tracks research questions and candidate experiments for
  [issue #684](https://github.com/gpsaggese/gpsaggese.github.io/issues/684)
- It is a backlog, not a literature review, experimental protocol, implementation
  specification, or record of findings
- The items below are ideas to investigate and propose for review; they are not
  accepted Noesis design decisions

## Work Completed Before This Research

- `PR_S11` established the reusable probing foundation:
  - A versioned 300-question candidate canary bank
  - Deterministic normalization and grading
  - Pluggable capability measurement
  - Calibration and activation tools
  - Secret, balanced canary scheduling for active contracts
  - Live-provider probe execution and PostgreSQL evidence storage
- [PR #737](https://github.com/gpsaggese/gpsaggese.github.io/pull/737) proposes
  the independent-monitoring boundary and splits the work into reviewable stages:
  design, one-shot execution, recurring windows, and degradation detection
- [PR #739](https://github.com/gpsaggese/gpsaggese.github.io/pull/739) implements
  the first standalone one-shot monitoring runtime with monitoring-only persistence,
  bounded execution, explicit paid-call authorization, and fixture-based tests
- The standalone path does not create market orders or contracts and cannot write
  contract verdicts or seller reputation

## Questions to Research

### Define the Claim

- What observable behavior should Noesis call model degradation or "nerfing"?
- How should the system distinguish a sustained degradation from normal sampling
  variation?
- Can the monitor detect a behavioral change without claiming why the provider made
  it?
- Which claims can be supported by deterministic probes, and which require a broader
  evaluation method?

### Understand Existing Approaches

- Review work on continuous LLM evaluation, behavioral drift, change-point
  detection, sequential testing, and black-box model monitoring
- Compare deterministic known-answer probes with:
  - Agreement with a stronger reference model
  - LLM-as-judge evaluation
  - Public benchmark subsets
  - Multiple-model agreement
  - Human review
  - Hosted third-party evaluation services
- Record the cost, latency, reproducibility, false-positive risk, and provider-gaming
  risk of each approach
- Identify approaches that can complement the existing deterministic grader without
  replacing it

### Choose Detection Methods to Compare

- Evaluate the proposed one-sided Fisher exact test as an initial baseline
- Investigate sequential probability tests and statistical process-control methods
- Investigate change-point methods for abrupt and gradual degradation
- Determine how to account for repeated testing across many windows and targets
- Determine whether an alert should require:
  - Statistical significance
  - A minimum practical quality drop
  - A minimum sample count
  - Multiple consecutive degraded windows
- Measure detection delay and false-alarm rate, not only whether a method eventually
  detects a change

### Control Important Confounders

- Time-of-day and day-of-week effects
- Provider outages and transient rate limiting
- Changes in provider routing or served-provider attribution
- Announced model releases and undisclosed model swaps
- Changes to model identifiers, provider aliases, or quantization
- Probe-bank revisions and question leakage
- Prompt-format, temperature, token-budget, and timeout changes
- Price changes that affect cost but not quality

## Candidate Experiments to Propose

These are possible experiments for later review, not an approved execution plan.

### Controlled Model-swap Experiment

- Establish a stable baseline using a stronger model
- Replace it at a known time with a weaker model while keeping the provider interface
  and probe distribution fixed
- Measure whether each candidate detector identifies the change, how quickly it does
  so, and whether it produces false alarms before the swap

### Latency-only and Availability-only Experiments

- Inject latency without changing answer quality
- Inject timeouts or upstream failures without changing successful answers
- Confirm that quality, latency, and availability remain separate signals

### Price-only Experiment

- Change the reported task cost while holding responses fixed
- Confirm that cost metrics change without producing a model-quality degradation
  verdict

### Time-of-day Study

- Probe the same provider/model combination in matched windows across the day
- Repeat the schedule over multiple days
- Compare matched time windows so that daily load patterns are not mistaken for a
  release or model swap

### Release and Model-version Study

- Record known provider and model release dates alongside the measurement series
- Start a new baseline when the model identity or probe-set version changes
- Investigate how to handle a provider alias whose backing model changes without a
  new public identifier

### Evaluation-method Comparison

- Grade the same response set with deterministic answers and one alternative method
- Compare agreement, disagreements, cost, latency, and reproducibility
- Manually inspect a small disagreement sample before deciding whether another
  grader adds useful evidence

### Probe-bank Robustness Study

- Compare results by canary family rather than only aggregate accuracy
- Test whether conclusions survive rotating to a new bank version
- Investigate private generated variants that preserve deterministic grading while
  reducing memorization risk

## Measurements to Preserve

- Exact provider, model identifier, probe-set version, and runtime configuration
- Timestamp and planned window identity
- Attempted calls, returned calls, and deterministically graded calls
- Correct and incorrect grades by canary family
- Availability and provider-status counts
- Latency distribution, including p50 and p95
- Prompt and completion token usage
- Cost per attempted task and cost per correct task
- Baseline identity, detector version, thresholds, and resulting decision
- Known release, incident, configuration-change, and controlled-intervention markers

## Evidence Eventually Needed

- A reviewed paid calibration run establishing which probes distinguish the healthy
  and weaker models
- Healthy fixture and live-provider runs that estimate the false-alarm rate
- Controlled degraded runs that estimate detection delay and sensitivity
- Reproducible tables and graphs for quality, availability, latency, and cost over
  time
- A comparison of the initial detector with at least one credible alternative
- A written account of limitations, confounders, and claims that the collected data
  cannot support

## Questions for Review

- Which provider/model targets are safe and useful for the first live study?
- What time horizon and call budget are acceptable?
- Should the first comparison prioritize a sequential test, a change-point method,
  or an alternative grader?
- What practical quality drop is large enough to matter?
- How should announced releases and undisclosed provider changes affect baselines?
- Which graphs and summary statistics are required for the probing paper?

## Deferred Work

- Collecting and summarizing related-work sources
- Selecting the final experiment protocol and statistical thresholds
- Running paid or long-lived provider experiments
- Making empirical claims about provider degradation or intent
- Connecting standalone monitoring results to market decisions

