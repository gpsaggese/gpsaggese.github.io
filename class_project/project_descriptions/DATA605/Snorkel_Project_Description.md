# Description

Snorkel is an open-source framework for programmatic labeling with weak supervision.
It solves the problem of the cost of hand-labeling a training set: the user writes
labeling functions, and a label model combines their noisy and conflicting votes into
probabilistic labels. It is worth a 60-minute tutorial because a labeled dataset of
tens of thousands of examples is built in an afternoon, and the quality of the labels
can be measured on a small dev set.

## Technologies Used

Snorkel

- Labeling functions written with `@labeling_function()`, which can `ABSTAIN`
- `PandasLFApplier` and `LFAnalysis` to measure coverage, overlaps, and conflicts
- `LabelModel` and `MajorityLabelVoter` to aggregate the votes into labels
- Slicing and transformation functions to monitor subsets and augment data

# Tutorial

- Implement the tutorial "Learn Snorkel in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of `.claude/skills/tutorial_in_60_mins.rules.md`
  - No earlier tutorial or project uses Snorkel, so read the closest text
    classification work
  - Read the `README.md` of the Spring2026 FastText and HuggingFace text
    classification projects, and reuse what is good
    - `class_project/data605/Spring2026/projects/UmdTask458_DATA605_Spring2026_FastText_text_classification/`
    - `class_project/data605/Spring2026/projects/UmdTask443_DATA605_Spring2026_HuggingFace_Text_Classification_Model/`
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare briefly with modAL and small-text from the labeling cost point of view,
  i.e., weak supervision vs. active learning
- Deliverables:
  - `snorkel_utils.py`
  - `snorkel.API.ipynb`
  - `snorkel.example.ipynb`

# Project

## Project 1 (Fall2026): Financial News Sentiment with Weak Labels

- **Project Objective**: Classify finance tweets as bearish, bullish, or neutral without
  labeling the training set by hand, using labeling functions and a label model
- **Dataset Suggestions**:
  [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
- **Tasks**:
  - **Preprocess the Text**: Load the train and validation files, hold out 500 labeled
    tweets from the train file as the dev set, ignore the labels of the other train
    tweets, and use the validation file as the test set
  - **Define the Problem**: Fix the three classes and macro F1 as the metric, measure
    the majority-class baseline, and tune the functions on the dev set only
  - **Write Labeling Functions**: Write 10-15 keyword and regex functions with
    `@labeling_function()`, e.g., "beats estimates", "downgrade", or a signed percent
    move such as "-4%", and report coverage, overlaps, and conflicts with `LFAnalysis`
  - **Train the Models**: Fit `LabelModel` and `MajorityLabelVoter`, and train a
    logistic regression on TF-IDF features with the probabilistic labels
  - **Evaluate the Models**: Report accuracy and macro F1 on the test set for the
    majority vote, the label model, the end model, and a fully supervised upper bound
  - **Analyze the Errors**: Plot the accuracy and coverage of each function, inspect
    the tweets that confuse bullish and bearish, and refine two functions
- **Bonus Ideas (Optional)**: Add a labeling function from the
  [Loughran-McDonald lexicon](https://sraf.nd.edu/loughranmcdonald-master-dictionary/);
  test the same functions on
  [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Preprocess the Text
  - Result: project dir created and container running, and the counts of the train,
    dev, and test splits
- Milestone 2: API notebook
  - Project tasks: Write Labeling Functions, Train the Models
  - Result: `snorkel.API.ipynb` covering `labeling_function`, `PandasLFApplier`,
    `LFAnalysis`, `MajorityLabelVoter`, and `LabelModel`
- Milestone 3: Example notebook
  - Project tasks: Define the Problem, Write Labeling Functions, Train the Models,
    Evaluate the Models, Analyze the Errors
  - Result: `snorkel.example.ipynb` running end to end

## Project 2: Routing Consumer Complaints with Augmentation

- **Project Objective**: Route bank customer complaints to the right financial product
  with weak labels, and test if data augmentation improves the end model
- **Dataset Suggestions**:
  [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/),
  using 20,000 complaints with a narrative from 2023 onward
- **Tasks**:
  - **Load the Complaints**: Read the CSV file, keep five products (credit reporting,
    debt collection, mortgage, credit card, and checking or savings account), and split
    by `Date received` so that dev and test hold the newest complaints
  - **Write Labeling Functions**: Write functions for product names, credit bureau
    names, and fee words, and one with a `preprocessor` that adds a lowercase text
    field without the `XXXX` redaction tokens
  - **Aggregate the Labels**: Compare `LabelModel` and `MajorityLabelVoter` on the dev
    set with precision, recall, and macro F1, against the majority-class baseline
  - **Augment the Data**: Write two `transformation_function`s, e.g., a swap with a
    small finance synonym list and random deletion, and apply them with
    `PandasTFApplier` and a `RandomPolicy`
  - **Evaluate the End Model**: Train logistic regression with and without the
    augmented data, and compare the macro F1 on the test set
- **Bonus Ideas (Optional)**: Add a labeling function from a pre-trained zero-shot
  classifier; compare with training on 500 hand-labeled complaints

## Project 3: Robustness of Weak Supervision for Credit Default

- **Project Objective**: Measure how noisy labeling functions degrade the label model
  and the majority vote on an imbalanced credit risk task, and find the borrower slices
  where the end model fails
- **Dataset Suggestions**:
  [UCI - Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Tasks**:
  - **Load the Clients**: Read the 30,000 clients, and make a stratified split into an
    unlabeled train set, a labeled dev set of 1,000 clients, and a test set of 6,000
    clients, keeping the 22% default rate in each
  - **Write Diverse Functions**: Write 8-10 functions on the repayment status `PAY_0`
    to `PAY_6`, the credit utilization `BILL_AMT1 / LIMIT_BAL`, and the ratio of the
    payment to the bill, without using `SEX` or `MARRIAGE`
  - **Inject Noise**: Add 0, 2, 4, and 6 labeling functions that vote at random
  - **Compare the Aggregators**: For each noise level, fit `LabelModel` with the
    `class_balance` of the dev set and `MajorityLabelVoter`, record the F1 of the
    default class on the dev set, and compare with a logistic regression trained on
    the 1,000 dev labels
  - **Evaluate on Slices**: Define three `slicing_function`s, e.g., credit limit below
    50,000, age under 30, and utilization above 90%, and report the end model F1 with
    `slice_dataframe`
  - **Report the Robustness**: Plot the end model F1 of the default class vs. the
    number of noisy functions for both aggregators, with the standard deviation over
    five seeds
- **Bonus Ideas (Optional)**: Estimate the accuracy of each function from the label
  model weights, and drop the worst functions
