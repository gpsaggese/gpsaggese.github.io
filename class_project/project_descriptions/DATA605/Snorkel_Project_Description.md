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
- Create `tutorials/Snorkel/`, since it does not exist yet
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

## Project 1: News Topic Classification with Weak Labels

- **Difficulty**: 1 (Easy)
- **Project Objective**: Classify news articles into four topics without labeling the
  training set by hand, using labeling functions and a label model
- **Dataset Suggestions**:
  [AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- **Tasks**:
  - **Preprocess the Text**: Merge the title and the description, and split the data
    into an unlabeled train set, a labeled dev set of 500 articles, and a test set
  - **Define the Problem**: Fix the four classes and macro F1 as the metric, and
    measure the majority-class baseline
  - **Write Labeling Functions**: Write 10-15 keyword and regex functions with
    `@labeling_function()`, and report coverage, overlaps, and conflicts with
    `LFAnalysis`
  - **Train the Models**: Fit `LabelModel` and `MajorityLabelVoter`, and train a
    logistic regression on TF-IDF features with the probabilistic labels
  - **Evaluate the Models**: Report accuracy and macro F1 on the test set for the
    majority vote, the label model, the end model, and a fully supervised upper bound
  - **Analyze the Errors**: Plot the accuracy and coverage of each function, inspect
    the misclassified articles, and refine two functions
- **Bonus Ideas (Optional)**: Add a labeling function that wraps a zero-shot
  classifier; compare with training on 200 hand-labeled articles

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Preprocess the Text
  - Result: `tutorials/Snorkel/` container running, and the counts of the train, dev,
    and test splits
- Milestone 2: API notebook
  - Project tasks: Write Labeling Functions, Train the Models
  - Result: `snorkel.API.ipynb` covering `labeling_function`, `PandasLFApplier`,
    `LFAnalysis`, `MajorityLabelVoter`, and `LabelModel`
- Milestone 3: Example notebook
  - Project tasks: Define the Problem, Write Labeling Functions, Train the Models,
    Evaluate the Models, Analyze the Errors
  - Result: `snorkel.example.ipynb` running end to end

## Project 2: Spam Detection in Comments with Augmentation

- **Difficulty**: 2 (Medium)
- **Project Objective**: Detect spam in video comments with weak labels, and test if
  data augmentation improves the end model
- **Dataset Suggestions**:
  [UCI - YouTube Spam Collection](https://archive.ics.uci.edu/dataset/380/youtube+spam+collection)
- **Tasks**:
  - **Load the Comments**: Merge the five video files and split them into train, dev,
    and test sets
  - **Write Labeling Functions**: Write functions for links, calls to action, and
    comment length, and one with a `preprocessor` that adds a lowercase text field
  - **Aggregate the Labels**: Compare `LabelModel` and `MajorityLabelVoter` on the
    dev set with precision, recall, and F1
  - **Augment the Data**: Write two `transformation_function`s, e.g., synonym swap
    and random deletion, and apply them with `PandasTFApplier`
  - **Evaluate the End Model**: Train logistic regression with and without the
    augmented data, and compare the test F1
- **Bonus Ideas (Optional)**: Add a labeling function from a pre-trained sentiment
  model

## Project 3: Robustness of Weak Supervision on Newsgroups

- **Difficulty**: 3 (Hard)
- **Project Objective**: Measure how noisy labeling functions degrade the label model
  and the majority vote, and find the data slices where the end model fails
- **Dataset Suggestions**:
  [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset),
  using five categories
- **Tasks**:
  - **Load the Newsgroups**: Fetch five categories with `fetch_20newsgroups`,
    removing headers, footers, and quotes
  - **Write Diverse Functions**: Write keyword and regex functions, and a function
    that uses the vocabulary learned from a small labeled seed set
  - **Inject Noise**: Add 0, 2, 4, and 6 labeling functions that vote at random
  - **Compare the Aggregators**: For each noise level, fit `LabelModel` and
    `MajorityLabelVoter`, and record the accuracy of the labels on the dev set
  - **Evaluate on Slices**: Define three `slicing_function`s, e.g., short documents
    and documents with conflicting votes, and report the end model F1 with
    `slice_dataframe`
  - **Report the Robustness**: Plot the end model macro F1 vs. the number of noisy
    functions for both aggregators, with the standard deviation over five seeds
- **Bonus Ideas (Optional)**: Estimate the accuracy of each function from the label
  model weights, and drop the worst functions
