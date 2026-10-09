# Description

W&B Weave is a toolkit from Weights & Biases to build, trace, and evaluate
applications that use models, in particular LLM applications. It solves the problem
of not knowing what a pipeline did and whether a change made it better: every call of
an instrumented function is logged as a trace, and datasets, models, and evaluations
are versioned and compared in a web UI. It is worth a 60-minute tutorial because
adding `weave.init` and `@weave.op` to existing code gives full traces and comparable
evaluations at almost no cost.

## Technologies Used

W&B Weave

- Tracing of function calls with `weave.init` and `@weave.op`
- Versioned datasets and models (`weave.Dataset`, `weave.Model`)
- Evaluations with scorers (`weave.Evaluation`) to compare versions of a pipeline
- Web UI to inspect traces, compare evaluations, and share results with a team

# Tutorial

- Implement the tutorial "Learn W&B Weave in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - No earlier tutorial or project uses Weave, so read the Fall2025 Weights & Biases
    project for the related tracking tool, and reuse what is good
    - `class_project/msml610/Fall2025/projects/TutorTask_103_Weights_and_Biases_Hard/`
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `weave_utils.py`
  - `weave.API.ipynb`
  - `weave.example.ipynb`

# Project

## Project 1 (Fall2026): Tracing and Evaluating Hawkish-Dovish Models for FOMC Text

- **Project Objective**: Build tone models that label the hawkish or dovish stance of
  Federal Reserve text, use Weave to trace them and to compare them with
  `weave.Evaluation` on labeled sentences, and test whether the tone of the FOMC
  statements relates to the 2-year Treasury yield change on the announcement day
- **Dataset Suggestions**: Labeled central bank text, FOMC statements, and yields
  - [Trillion Dollar Words: FOMC Hawkish-Dovish Sentences](https://huggingface.co/datasets/gtfintechlab/fomc_communication)
    with `train.csv` and `test.csv` (labels: dovish, hawkish, neutral)
  - [FOMC Statements](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
    from the Federal Reserve website (the calendar links each statement page)
  - [Daily Treasury Par Yield Curve Rates](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2024)
    for the 2-year yield (one page and one CSV per year)
- **Tasks**:
  - **Ingest the Data**: Load the labeled sentences, download the FOMC statements of
    2015-2024 and the 2-year yields, cache them on disk, and compute the yield change
    in basis points from the previous close to the close of the statement day
  - **Define the Problem**: Classify sentences as hawkish, dovish, or neutral with
    macro-F1 on `test.csv`, using a TF-IDF and logistic regression baseline fitted on
    `train.csv`, and define the tone of a statement as the share of hawkish minus the
    share of dovish sentences
  - **Trace the Models**: Implement the baseline, the pre-trained `FOMC-RoBERTa`
    classifier, and a zero-shot classifier as `weave.Model` classes, with `predict`
    decorated by `@weave.op` after `weave.init`, and inspect the traces
  - **Evaluate with Weave**: Publish the `test.csv` sentences as a `weave.Dataset`,
    and run `weave.Evaluation` for each model with an accuracy scorer and a custom
    `weave.Scorer` that reports macro-F1 in its `summarize` method
  - **Link Tone to Yields**: Pick the model by macro-F1 on the sentences and never by
    the yield correlation, then compute the Spearman correlation and a bootstrap 95%
    confidence interval between the change of tone and the yield change, and compare
    it with the same statistic for the baseline model
  - **Visualize and Report**: Compare the evaluations in the Weave UI, and plot the
    change of tone against the yield change with a fitted line
- **Bonus Ideas (Optional)**: Add a hosted LLM as a fourth tone model, and compare its
  macro-F1, latency, and cost with the other models in the Weave UI

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Ingest the Data
  - Result: project dir created and container running with a Weave project, and a
    cached table with the labeled sentences, the FOMC statements, and the 2-year
    yield change by statement date
- Milestone 2: API notebook
  - Project tasks: Trace the Models, Evaluate with Weave
  - Result: `weave.API.ipynb` covering `weave.init`, `@weave.op`, `weave.Dataset`,
    `weave.Model`, and `weave.Evaluation` on a toy classification function
- Milestone 3: Example notebook
  - Project tasks: Define the Problem, Trace the Models, Evaluate with Weave, Link
    Tone to Yields, Visualize and Report
  - Result: `weave.example.ipynb` running end to end

## Project 2: Credit Default Scoring with Versioned Models

- **Project Objective**: Build a credit card default scoring pipeline, and use Weave to
  version the data and the models and to compare a logistic regression baseline with a
  gradient boosting model on discrimination and on customer segments
- **Dataset Suggestions**:
  [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
  (30,000 clients of a Taiwanese bank, with the default flag of the next month)
- **Tasks**:
  - **Load the Data**: Import the 30,000 clients, rename the target to `default`, and
    hold out a stratified 20% test set with a fixed seed, since the data has no date
  - **Engineer Features**: Create the credit utilization (bill amount over the credit
    limit), the payment ratio, and the number of months with a payment delay
  - **Trace the Pipeline**: Wrap the preprocessing and the prediction in `@weave.op`
    after `weave.init`, and publish the test table as a versioned `weave.Dataset`
  - **Train the Models**: Fit a logistic regression baseline and a
    `HistGradientBoostingClassifier`, each wrapped in a `weave.Model` with its
    hyperparameters as attributes
  - **Evaluate with Weave**: Run `weave.Evaluation` for both models with scorers for
    ROC AUC, the KS statistic, and the share of defaults caught when the 20% riskiest
    clients are rejected
  - **Audit the Segments**: Exclude `SEX` from the features, and compare the AUC and
    the approval rate of both models by sex, age band, and credit limit band
- **Bonus Ideas (Optional)**: Add a Brier score scorer to compare the calibration of
  the models; compute the expected loss of an approval cutoff in which a missed default
  costs five times a refused good client

## Project 3: Tracing a Retrieval Pipeline for Financial Questions

- **Project Objective**: Build a retriever that finds the answer passage of a financial
  opinion question, and use Weave to trace the pipeline and to compare a keyword
  baseline with a dense retriever by the quality of the ranking
- **Dataset Suggestions**:
  [FiQA-2018 in BEIR Format](https://huggingface.co/datasets/mteb/fiqa) (questions and
  answer passages about personal finance and investing)
  - Use `corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv`
  - Keep the relevant passages of the test queries plus 10,000 random passages, so
    that the corpus fits on a laptop
- **Tasks**:
  - **Load the Corpus**: Read the three files, and build the subset corpus, the test
    queries, and the relevance labels
  - **Build the Baseline**: Rank the passages with BM25 (`rank_bm25`), and compute
    hit@10 and MRR@10 on the test queries
  - **Build the Dense Retriever**: Embed the passages and the queries with a
    `sentence-transformers` model such as `all-MiniLM-L6-v2`, and rank by cosine
    similarity
  - **Trace the Pipeline**: Implement each retriever as a `weave.Model` whose `predict`
    calls `@weave.op` steps for the query embedding and the ranking, and inspect the
    trace tree and the latency
  - **Evaluate with Weave**: Publish the test queries as a `weave.Dataset`, and run
    `weave.Evaluation` for both models with scorers for hit@10 and the reciprocal rank
  - **Inspect the Failures**: Filter in the Weave UI the queries that the dense model
    misses, and label 20 of them by cause, such as vocabulary mismatch or vague
    question
- **Bonus Ideas (Optional)**: Fuse the two rankings with reciprocal rank fusion and
  compare it with each retriever; add an LLM-judge scorer for the relevance of the top
  passage
