---
title: "We Broke Our Data on Purpose. Here's Who Noticed."
description: "What a reviewer agent catches in data science pipelines: it did not make ours more accurate, but it caught broken data and leaked features that other systems missed."
authors:
  - Under-the-stars
  - linuteresa
  - gpsaggese
date: 2026-11-15
categories:
  - AI Research
draft: true
---

**TL;DR:** A reviewer agent did not make our data science pipeline more accurate. It made it safer. It found every impossible value we planted. It flagged a leaked column in 10 of 10 runs, where the other systems, AutoGluon included, reported a perfect score of 1.0 that meant nothing. The honest score was 0.65. It also dropped a real leak in the taxi data that we had not planted, in 10 of 10 runs. It missed a second, weaker real leak, so it is a safety net with holes, not a guarantee. The agent is not smarter. It is more careful.

<!-- more -->

Our multi-agent pipeline beat AutoGluon at predicting heart disease, 88% F1 against 73%. We were ready to write it up as a win.

Then we looked at the test set. It had 61 rows. The two systems had used different splits, and we had picked the winning model by F1 on that same test set. The win was not real, and we had to rerun the whole project to find out what was.

## Why this matters

This matters because agent frameworks promise reliability, and most write-ups prove it with one impressive score. A score tells you how a system did on one dataset on one day. Reliability is different. It means the system notices when something is wrong with its inputs and says so before it reports a number you might act on. That second property is what we set out to measure, and it turned out to be harder to measure than accuracy.

## What we tested

The question: does an AI pipeline with a built-in quality checker give more reliable results than a plain one-pass pipeline, especially when the data is messy?

We built the pipeline as a [LangGraph](https://www.langchain.com/langgraph) graph with five steps: plan, explore the data, train models, review, write a report. The review step is the interesting one. It checks the work and can send the graph back to an earlier step instead of moving on.

We compared it with a single-pass version, with [AutoGluon](https://auto.gluon.ai), and with a model that predicts "yes" for everyone. AutoGluon ran with a 60 second limit and the medium quality preset. Every fit finished in under 31 seconds, and a 300 second limit gave identical scores. Every system got the same train/test split, the same corrupted training data and the same clean test set, on three datasets:

- **Heart disease:** the [Cleveland data](https://archive.ics.uci.edu/dataset/45/heart+disease), 303 patients. Tiny and easy to overfit.
- **NYC taxi tips:** a sample of [yellow cab trips](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Whether a ride was tipped.
- **Amazon Beauty reviews:** a sample of [Amazon reviews](https://amazon-reviews-2023.github.io/). Whether a review is positive.

Then we damaged the training data on purpose: random wrong labels (5%, 15% and 30%), impossible values like an age of -9999, and one fake column that leaks the answer. The test data always stayed clean.

## How the pipeline works

Each of the five steps has one job.

- **Planner:** looks at the data and writes down the risks, such as unbalanced classes or missing values. A language model can add a one-line plan, but nothing depends on it.
- **EDA analyst:** fills in missing values using the training rows only, then applies any fixes the reviewer asks for, such as dropping a row or a column.
- **ML architect:** tries three standard models (random forest, logistic regression and gradient boosting), picks the best one by cross-validated MCC on the training data, and scores it on the held-out test set.
- **Quality reviewer:** the checker. It decides whether the result can be trusted and, if not, where to send the graph back to.
- **Report writer:** records the plan, what was fixed, the metrics and the full log.

```mermaid
flowchart LR
  A[Planner] --> B[EDA Analyst]
  B --> C[ML Architect]
  C --> D{Quality Reviewer}
  D -- pass --> E[Report Writer]
  D -- retry_eda --> B
  D -- retry_ml --> C
```

In one line: Planner → EDA Analyst → ML Architect → Quality Reviewer → Report Writer, where the reviewer can send the graph back to the EDA Analyst or the ML Architect.

The two arrows that go back are the point. A plain pipeline runs once and reports whatever comes out. This one can stop, fix the data and try again, up to three times.

**What the reviewer checks**

1. **Impossible values.** An age of -9999 or a negative fare. The reviewer has hand-written limits for each dataset, and it sends the graph back to drop those rows.
2. **A leaked feature.** One column that almost gives away the answer on its own. If dropping a single column makes a suspiciously high score collapse, that column is the leak.
3. **Wrong labels.** Rows where a model trained on the rest of the data strongly disagrees with the given label.
4. **A model that just says yes.** If almost every prediction is positive, or recall and specificity are badly lopsided, the reviewer retries with class weights, a tuned threshold or a different model.

If a retry changes nothing, the graph stops and says so instead of looping.

## Why our first result was wrong

Three things went wrong at once:

- **Different splits and models.** Each system got its own train/test split and its own models, so a score gap could come from anywhere.
- **A 61-row test set.** One extra correct prediction moves the score by about 1.6 points.
- **F1 as the judge.** F1 rewards a model that says "positive" for everyone, and we were also choosing models by F1 on the test set.

The F1 problem is the easiest to see. About 80% of NYC taxi rides are tipped, so a model that always answers "tipped" looks great:

| System (NYC taxi) | F1 | MCC |
|---|---|---|
| Always answers "tipped" | 0.889 | 0.000 |
| Real model | 0.894 | 0.202 |

By F1 they are nearly tied. By [MCC](https://en.wikipedia.org/wiki/Phi_coefficient), a score where 0 means guessing and 1 means perfect, the do-nothing model gets exactly 0. So we rebuilt the benchmark with one split for every system, models chosen only on training data, seeded corruption, many random seeds, and MCC as the main score.

## What the fair test shows

On accuracy, nobody clearly wins. Here are all three systems side by side, scored with MCC on the clean test set and averaged over 10 random splits per dataset. "LangGraph + reviewer" is our rebuilt multi-agent pipeline:

| Dataset | Wrong labels in training data | AutoGluon | Single pass | LangGraph + reviewer |
|---|---|---|---|---|
| Heart disease | 0% | 0.642 | 0.646 | 0.650 |
| Heart disease | 5% | 0.623 | 0.654 | 0.654 |
| Heart disease | 15% | 0.605 | 0.651 | 0.628 |
| Heart disease | 30% | 0.468 | 0.504 | 0.466 |
| Amazon reviews | 0% | 0.589 | 0.597 | 0.597 |
| Amazon reviews | 5% | 0.604 | 0.587 | 0.587 |
| Amazon reviews | 15% | 0.562 | 0.549 | 0.553 |
| Amazon reviews | 30% | 0.518 | 0.527 | 0.527 |
| NYC taxi tips | 0% | 0.159 | 0.221 | 0.124 |
| NYC taxi tips | 5% | 0.117 | 0.222 | 0.207 |
| NYC taxi tips | 15% | 0.112 | 0.222 | 0.208 |
| NYC taxi tips | 30% | 0.087 | 0.222 | 0.141 |

**How to read the table.**

- **Heart disease: a tie.** The three systems are within a few hundredths of each other at every noise level. The score moves by about 0.1 from one random split to the next, so gaps this small are noise. The reviewer ends up 0.02 to 0.04 below the single pass at 15% and 30% wrong labels, which is inside the noise too. All three fall as more labels go wrong, and the fall is steepest at 30%.
- **Amazon reviews: also a tie.** The reviewer and the single pass are within 0.004 of each other at every noise level. AutoGluon is level with them, except at 5% wrong labels where it is 0.017 higher, a small gap that is outside the noise.
- **NYC taxi tips: AutoGluon is lowest at every noise level**, by 0.06 to 0.14 MCC compared with the single pass, and with 10 splits that gap is clearly outside the noise. Taxi is a hard task (every MCC is 0.22 or below), so the numbers are small, but the ordering is the same at every noise level.
- **The reviewer costs a little on taxi.** Compared with the single pass it is lower on clean data (by 0.097) and at 30% wrong labels (by 0.081), and within the noise at 5% and 15%. Its retry step pushes the model toward a more balanced one. On clean data the single pass says "tipped" for every ride (recall 1.00, specificity 0.062), which is why its F1 matches the do-nothing model. AutoGluon does almost the same (recall 0.975, specificity 0.096). The reviewer's model catches more of the untipped rides (specificity 0.356, recall 0.767). That raised balanced accuracy (0.562 against 0.531) but lowered MCC and F1.
- **Nothing here says the multi-agent system is more accurate.** If you only wanted the best score, the single pass is as good or better everywhere.

Here is the same comparison for heart disease, split by reviewer version:

| Wrong labels in training data | Single pass | Reviewer (first version) | Reviewer (rebuilt) |
|---|---|---|---|
| 0% | 0.646 | 0.638 | 0.650 |
| 15% | 0.651 | 0.562 | 0.628 |
| 30% | 0.504 | 0.443 | 0.466 |

![Left: heart disease MCC by amount of label noise for the single pass, the first reviewer and the rebuilt reviewer. Right: reported MCC when a leaked column is added.](./draft.benchmarking_in_Agentic_Reasoning_for_Data_Science.md.figs/reviewer_accuracy_vs_leak.png)

**How to read the chart.** On the left, the three systems score about the same when the data is clean or lightly damaged, and all of them drop when 30% of the labels are wrong. The first reviewer (orange) is lowest at 15% and 30% noise, because it deleted good rows. The rebuilt reviewer (blue) recovers most of that loss but does not beat the single pass (grey). On the right, a column that leaks the answer pushes the single pass to a perfect 1.00. AutoGluon also scored 1.00 in all 10 splits (not drawn). The reviewers flag the column and report 0.65, which is the honest score. The point of the reviewer is the right panel, not the left.

Our first reviewer made things worse. It deleted rows it thought were mislabeled, and with only about 240 training rows it threw away good data. We rebuilt it so it removes rows only if a leak-free cross-validation test shows that helps. That fixed most of the damage, but it still does not beat the single pass. On Amazon reviews all systems tie. On taxi tips the rebuilt reviewer scores lower on MCC, because it forces a more balanced model.

**Where the reviewer earns its place**

- **Impossible values:** it found all of them, on all three datasets.
- **Planted leaks:** with a fake leaked column added, the single pass, the plain graph and AutoGluon scored a perfect MCC of 1.0. The reviewer flagged the column in 10 of 10 runs and reported the true 0.65.
- **No false alarms:** on clean data with no leak, the reviewer dropped no feature in any of 30 runs (10 splits on each dataset).
- **A natural leak it caught:** in the taxi data, the payment type almost gives away the tip, since cash tips are not recorded. We kept that column in and ran 10 splits. The single pass, the plain graph, the first reviewer and AutoGluon all scored an MCC of about 0.91. The rebuilt reviewer dropped the column in 10 of 10 splits and reported 0.106, which is the honest score for this task and close to the 0.124 it scores on clean data without that column. The first reviewer never dropped it.
- **A natural leak it missed:** we also kept the total fare amount, which includes the tip. Every system scored about 0.75 (AutoGluon 0.757), and the reviewer dropped the column in 0 of 10 splits. Our leak checks only fire when one column alone gets close to a perfect score, and this one does not. So the reviewer catches a strong leak and misses a weaker one. Both leaks are in one dataset.

By default its routing is rule-based: the language model only writes the plan and the summary. We also let a small local model (llama3.2) pick the retry strategy instead of the fixed order. This time we recorded every raw answer, over 10 taxi splits at two noise levels. The model was asked to pick a strategy 29 times. Every answer named a valid strategy (one was the misspelling "balance_class_weights", which we counted as class_weight), and in all 29 cases it picked the first strategy on the list, the same one the rules would pick. In the other runs the retries ended before the model was asked. So the language model never made a different choice from the rules. Its MCC was identical to the rules-based run on all 20 runs (mean 0.166 for both), so it changed no scores.

**A closer look at the label checks**

The reviewer could often spot suspicious labels. On the Amazon reviews, the rows it flagged were mislabeled about 60% of the time, and it found about 54% of the wrong labels we had planted. But removing the flagged rows never improved the final score. About 40% of what it flagged was fine, and the models cope with the rest well enough anyway. Spotting a problem and fixing it are two different skills.

Building the rebuilt check also taught us something about our own testing. Our first version of the "does removing rows help?" test looked great, even on clean data. It was fooled. The detector that picked the rows to drop had already seen the validation rows, so cleaning looked helpful when it was not. The fix was to repeat the whole detection step inside every cross-validation fold, so nothing from the validation rows leaks in. After that, on clean data the reviewer scored exactly the same as having no reviewer, which is the right answer.

**Did we tune the reviewer to look good on heart disease?**

We rebuilt the reviewer using heart disease, so we checked how much its settings matter. We ran six versions on 10 splits: a stricter and a looser label-removal test, shifted cutoffs for the label check, and stricter and looser leak checks. Compared with the single pass, none of them was more accurate. With clean data every version was within 0.003 MCC of the single pass. At 15% wrong labels the gap was 0.011 to 0.032 below, and at 30% it was 0.029 to 0.038 below. Only the looser label test removed any rows (7.1 per run on average), and it narrowed the gap a little without closing it. The leak check did not depend on the settings either. Every version caught the planted leak in 10 of 10 runs, and none raised a false alarm on clean data. So there was no setting that made the reviewer look better on heart disease, and the results on the other two datasets did not need any tuning.

## What to take from this

If you test an AI pipeline, a few habits would have saved us a rerun:

1. **Add a do-nothing baseline.** If "always yes" scores nearly as well, your metric is hiding something.
2. **Give every system the same split.** Choose models on training data only, and never on the test set.
3. **Plant faults and see who notices.** A pipeline that reports a perfect score on leaked data is the one to worry about.
4. **Report the null result.** "No accuracy gain, but it catches broken data" is less exciting than a win, and much more useful.

The limits are real. Each dataset used 10 random splits, and Amazon and taxi were 5,000-row samples. The table uses AutoGluon's medium preset. We also tried its best-quality preset with a 180 second limit, on 5 heart disease splits and 2 splits each of taxi and Amazon, at 0% and 30% wrong labels, while another job was running on the same laptop. It did not change the picture. On Amazon all three systems stayed within 0.02. On heart disease it matched the single pass (0.558 at 30% wrong labels, against 0.457 for the medium preset). On taxi it was a little higher than the medium preset (0.114 and 0.054 against 0.084 and 0.048) but still well below the single pass (0.190). That is a small set of splits, so we only use it to check that the preset is not what drives the result. The language model was one small local model (llama3.2). The planted leak is easy to spot, both natural leaks come from one dataset, and the reviewer missed the weaker one.

The code, the results and a step-by-step write-up are in the [project folder](https://github.com/gpsaggese/umd_classes/tree/UmdTask919_Blog_Benchmarking_in_agentic_Reasoning_for_Data_science/multi-agent-ds-benchmark). If you have run a similar comparison, or can see a flaw in ours, we would like to hear about it in the comments.
