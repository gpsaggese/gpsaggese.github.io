---
title: "Comparing Data-Science Agents on Benchmarks with Known Answers: Harness and First Results"
author:
  - name: Dev Shah + Giacinto Paolo (GP) Saggese
    department: "Department of Computer Science"
    organization: "University of Maryland, College Park"
    location: "College Park, MD, USA"
    email: "`dnts5203@umd.edu, gsaggese@umd.edu`{=typst}"
abstract: |
  Agents that write and run code are now routinely applied to data analysis,
  but it is hard to say how well they work: each agent is reported on a
  different benchmark, with its own prompt, model, and scoring, and many
  data-science benchmarks are hard to reproduce. Before building our own
  data-science agent, we need a fair way to measure the existing ones.
  <!-- -->
  This paper describes a harness that runs any agent on four public tabular
  benchmarks with known answers (InfiAgent-DABench, QRData, DA-Code, and
  DABstep), gives every agent the same prompt, the same model, and a private
  working copy of the data, and scores each answer with the benchmark's own
  published rules. A fixed sample of 160 tasks, a check that every gold answer
  scores 1, and per-run logs of cost, time, and errors make the numbers
  comparable and repeatable.
  <!-- -->
  As a first result, a general-purpose coding agent, Claude Code, run headless
  with no data-science-specific scaffolding, answers 78% of DABench, 70% of
  QRData, 60% of the DABstep dev tasks, and 52% of the DA-Code text tasks, at
  about \$0.25 per task and \$40 in total. The other three agents of the
  comparison are not run yet; we report what the harness found about the
  benchmarks themselves, including a malformed label, a download link that no
  longer works, and hidden test answers, and how an account usage limit nearly
  turned 72 tasks into silent zeros.
keywords:
  - data-science agents
  - benchmarking
  - LLM agents
  - data analysis
  - reproducibility
bibliography: references.bib
---

# Introduction

Coding agents changed how software is written because their work can be
checked: a test suite says whether a change is right. Data analysis has no
test suite. A claim about a dataset, such as "the fare grows with distance" or
"the treatment has no effect," cannot be checked against the process that
generated the data, because that process is unknown. The long-term goal of
this project is an agent that does data science the way coding agents write
code, and the main obstacle is exactly this missing verification.

Before building such an agent, we need to know how well the existing ones
work, and today that is hard to tell. Each published data-science agent is
evaluated on a different benchmark, with its own prompt, its own model, and
often its own scoring script, so two reported accuracies are rarely
comparable. A comparison across benchmarks with the same setup for every
agent is the first step, and it is the goal of this milestone: understand and
try the state of the art.

We restrict the problem to **tabular data analysis**: a question in natural
language, one or more data files, and a short answer. Almost every
data-agent benchmark is of this form, and its answers can be checked
automatically, which is the property we need.

The contributions of this paper are:

- **A comparison harness** that runs any agent on four public benchmarks with
  the same prompt, model, data, and limits, and scores answers with each
  benchmark's own rules (Section III).
- **A fixed, verified task sample** of 160 tasks, with a check that every gold
  answer scores 1 and an empty answer scores 0 (Section III).
- **First results** for a general-purpose coding agent on all four benchmarks,
  with confidence intervals, cost, and time (Section IV).
- **Reproducibility findings** about the benchmarks and the harness, which
  affect anyone comparing agents this way (Section V).

Section II reviews the benchmarks and agents, Section III describes the
method, Section IV the results, Section V the reproducibility findings,
Section VI the limitations, and Section VII the next steps.

# Related Work

**Data-analysis benchmarks.** InfiAgent-DABench [@hu2024infiagent] asks
closed-form questions about single CSV files and checks every requested value
in a fixed `@name[value]` format. QRData [@liu2024qrdata] targets statistical
and causal reasoning with data, with numerical and multiple-choice answers.
DA-Code [@huang2024dacode] covers the full data-science workflow, from
wrangling to machine learning, and scores most tasks on the files the agent
writes. DABstep [@egg2025dabstep] asks multi-step questions over business
tables and documentation from a payments company and keeps the answers of its
450 test tasks hidden behind a leaderboard. Each benchmark is usually reported
on its own; this paper runs the same agent on all four with one setup.

**Data-science agents.** Agents differ mainly in how they structure the work.
Minimal code-acting agents, such as the smolagents `CodeAgent`
[@smolagents], write Python as their action. Specialized agents add planning
and verification loops, as DS-STAR does with an explicit plan-verify cycle
[@nam2025dsstar], or ship with a benchmark, as DA-Agent does with DA-Code
[@huang2024dacode]. General-purpose coding agents, such as Claude Code
[@claudecode], were not designed for data analysis but can read files and run
code in a working directory.

**Comparisons of agents.** AgenticDataBench [@agenticdatabench2026] compares
several agent harnesses on data tasks with the same model and reports that no
harness wins in every domain. We follow the same principle, one model for all
agents so that differences come from the agent, and add confidence intervals,
cost, and a check of every benchmark's scorer.

# Method

## Benchmarks and Task Sample

Table I lists the four benchmarks and the tasks used. Each benchmark is
loaded as a list of tasks with a question (including the answer format the
benchmark expects), the data files the agent may read, and the gold answer.

| Benchmark | Tasks used | What it tests | Scoring rule |
|:---|:---|:---|:---|
| DABench | 256 dev | Closed-form questions on one CSV | All requested values match |
| QRData | 411 | Statistical and causal reasoning | Number within 3%, or option letter |
| DA-Code | 71 text-answer | Data-science code | JSON fields; floats within 0.01 |
| DABstep | 10 dev | Multi-step business analysis | Official DABstep scorer |

: Benchmarks and tasks used. {#tbl:benchmarks}

Three choices restrict the tasks. DA-Code has 500 tasks, but only the 71
scored by `compare_text` have a text answer; the rest are scored on CSV files,
plots, or models the agent writes and are left for later work. DABstep publishes
answers only for its 10 dev tasks, so those are the only ones we can score
locally. InfiAgent-DABench question 734 is skipped because its label repeats an
answer name with two different values, so no answer can match it.

From each benchmark we draw a fixed random sample of 50 tasks (seed 0); DABstep
contributes all 10 of its dev tasks, for 160 tasks in total. The sample is a
committed list of task ids, so every agent runs on the same tasks.

## Scoring

Every answer is scored with the benchmark's own published rules, so our numbers
are comparable with the original papers. We reimplemented the InfiAgent-DABench,
QRData, and DA-Code rules from their evaluation scripts and vendored the
official DABstep scorer without changes, after checking two independent copies
of it against each other. Before any agent runs, a check confirms for every
sampled task that the data files exist, that the gold answer written as an agent
would write it scores 1, and that an empty answer scores 0. All 160 tasks pass.

## Agents and Fair Setup

The comparison has four agents, chosen to cover the main designs: a minimal
code-acting agent (A1, smolagents `CodeAgent`), an agent shipped with a
benchmark (A2, DA-Agent), a general-purpose coding agent (A3, Claude Code), and
a recent specialized agent with planning and verification (A4, OpenDsStar, an
open implementation of DS-STAR). This paper reports A3; A1 is implemented and
A2 and A4 are in progress (Section VII).

Every agent gets the same setup:

- **Same prompt.** The question, the list of data files, the allowed libraries
  (pandas, numpy, scipy, statsmodels, scikit-learn), and an instruction to end
  with `FINAL ANSWER: <answer>`. Only the last instruction changes for agents
  that return their answer through a tool.
- **Same model.** All agents use the model set in one environment variable.
  Claude Code ran in its default configuration, with Claude Sonnet 5
  (`claude-sonnet-5`) as its main model; it also made small background calls
  to Claude Haiku 4.5 (`claude-haiku-4-5-20251001`), as both appear in its
  usage log. The other agents will use Claude Sonnet 5.
- **Private data.** Each run gets a fresh working directory with copies of the
  task's files, so no run can see another run's work.
- **Same limits.** At most 30 turns, 10 minutes, and \$1 per task.
- **No extra context.** Claude Code runs headless through the Claude Agent SDK
  with user settings and project instructions disabled, so every run starts
  from the same state.

## Metrics

For each agent and benchmark we report accuracy with a 95% bootstrap confidence
interval (2,000 resamples), the number of runs that ended in an error, the
total cost in US dollars, and the mean wall-clock time per task. An error
counts as a wrong answer.

# Results

Table II and Figure 1 show the results of Claude Code on the 160 tasks.

| Benchmark | n | Acc. | 95% CI | Err. | \$ | s/task |
|:---|---:|---:|:---|---:|---:|---:|
| DABench | 50 | 78% | 66–88% | 0 | 10.85 | 39 |
| QRData | 50 | 70% | 56–82% | 0 | 13.13 | 88 |
| DABstep | 10 | 60% | 30–90% | 1 | 3.99 | 114 |
| DA-Code | 50 | 52% | 38–66% | 1 | 11.74 | 58 |

: Claude Code results: accuracy, runs ending in an error, total cost in US
dollars, and mean time per task. DABstep uses its dev tasks and DA-Code its
text-answer tasks. {#tbl:results}

![Accuracy of Claude Code per benchmark, with 95% bootstrap confidence
intervals.](figures/accuracy_by_benchmark.png){#fig:accuracy width=100%}

Three observations follow.

**A general-purpose agent is a strong baseline.** With no data-science
scaffolding, Claude Code answers between half and four fifths of the tasks.
Any specialized agent should be measured against this baseline, not against a
plain language model.

**Difficulty follows the shape of the task.** Accuracy is highest on
InfiAgent-DABench, whose questions concern one CSV file and specify the answer
format exactly, and lowest on DA-Code, whose tasks need more data preparation
before the answer. The confidence intervals of these two benchmarks only
meet at their ends, which suggests a real gap; the others overlap and cannot be
ranked with 50 tasks. DABstep, with only 10 tasks, has an interval 60 points wide and
should not be compared with the others.

**Cost is small and predictable.** The whole run cost about \$40, between
\$0.22 and \$0.40 per task, and took between 40 seconds and 2 minutes per task
on average. The cost of a full
four-agent comparison on this sample is therefore in the low hundreds of
dollars.

# Reproducibility Findings

Building the harness surfaced problems that affect any comparison of agents on
these benchmarks.

- **A malformed label.** InfiAgent-DABench question 734 lists the same answer
  name twice with different values. The official script turns the label into a
  dictionary, which silently keeps only the last value of each name, so the
  question is scored against part of its own answer.
- **Data that is hard to get.** The DA-Code GitHub repository holds the data of
  only 62 of its 500 tasks; the rest is a 3.4 GB archive on Google Drive. The
  download command in the DA-Code README now returns HTTP 403, and only the
  browser link in the same README still works.
- **Hidden answers.** DABstep keeps about 98% of its answers hidden, so a local
  evaluation is limited to 10 tasks unless results are submitted to its public
  leaderboard.
- **Infrastructure failures look like wrong answers.** In our first full run,
  the account behind Claude Code hit a usage limit after 38 QRData tasks. Every
  later call failed within two seconds, and 72 tasks were recorded as errors;
  without inspecting the errors, the result would have read 0% on DA-Code and
  DABstep. The harness now stops after five failures in a row and can rerun only
  the failed tasks, and the numbers above come from that rerun. Reported
  accuracy should always come with the count of runs that ended in an error.

# Discussion and Limitations

- **One agent so far.** The paper sets up a comparison of four agents but
  reports one. The results are a baseline, not a ranking.
- **Small samples.** With 50 tasks per benchmark, confidence intervals are
  about ±12 points; with 10 DABstep tasks, about ±30. Larger samples would
  narrow them at a proportional cost.
- **Partial benchmarks.** DA-Code is limited to its text-answer tasks and
  DABstep to its dev tasks, so neither result represents the whole benchmark.
- **One run per task.** Agents are not deterministic; repeated runs would show
  how much accuracy varies from run to run.
- **No failure analysis yet.** We have not yet classified why answers are
  wrong, for example format errors against wrong values. This analysis scopes
  the next milestone.

# Conclusion and Next Steps

A harness that runs every agent with the same prompt, model, data, and limits,
and scores each benchmark with its own rules, makes data-science agents
comparable. Its first result is that a general-purpose coding agent already
answers 52% to 78% of tasks on four benchmarks for about \$0.25 per task.

The next steps complete Milestone 1: run the smolagents `CodeAgent` (A1), add
DA-Agent (A2) and OpenDsStar (A4), run all four on the same 160 tasks with the
same model, and classify the wrong answers. Milestone 2 then targets what that
analysis shows the agents do not do properly.

The code is in `research/agentic_data_science/bakeoff/`. The results in this
paper can be reproduced with:

```
download_benchmarks.py --download
download_benchmarks.py --check
run_benchmarks.py --agents claude_code
run_benchmarks.py --summary
```

