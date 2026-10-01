# Description

`trl` (Transformer Reinforcement Learning) is the Hugging Face library to post-train
language models with supervised fine-tuning, reward modeling, preference
optimization, and reinforcement learning. It solves the problem of steering a
pre-trained language model towards a goal that is hard to write as a loss, e.g.,
style, politeness, or user preference, by optimizing a reward. It is worth a
60-minute tutorial because the same trainer classes cover the whole alignment recipe,
and small models such as GPT-2 show the effect on a laptop.

## Technologies Used

`trl`

- Supervised fine-tuning with `SFTTrainer`
- Reward modeling with `RewardTrainer` from preference pairs
- Preference optimization with `DPOTrainer`
- Reinforcement learning against a reward function with `GRPOTrainer` and
  `PPOTrainer`, with a KL penalty to the reference model

# Tutorial

- Implement the tutorial "Learn trl in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of `.claude/skills/tutorial_in_60_mins.rules.md`
  - Read the `README.md` of the Fall2025 `trl` projects, and reuse what is good
    - `class_project/msml610/Fall2025/projects/UmdTask_43_Fall2025_trl_Dialogue_System_Enhancement/`
    - `class_project/msml610/Fall2025/projects/UmdTask20_Fall2025_trl_Sentiment_Analysis_with_Reinforcement_Learning/`
  - Read the `README.md` of the Fall2025 `trlx` and PEFT projects for the related
    post-training tools
    - `class_project/msml610/Fall2025/projects/UmdTask18_Fall2025_trlx_Automated_Text_Summarization_with_Feedback_Loop/`
    - `class_project/msml610/Fall2025/projects/UmdTask96_Fall2025_PEFT_Sentiment_Analysis_on_Movie_Reviews/`
  - Read the notebooks of `msml610/tutorials/L12_reinforcement_learning/` for the
    reinforcement learning background
  - The `trl` API changes across versions, so pin the version in the Docker container
    and check that the Fall2025 code still runs
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare briefly with `trlx` from the training loop point of view, i.e., the reward
  function, the KL control, and the supported algorithms
- Deliverables:
  - `trl_utils.py`
  - `trl.API.ipynb`
  - `trl.example.ipynb`

# Project

## Project 1 (Fall2026): Financial News Sentiment with a Verifiable Reward

- **Project Objective**: Teach a small language model to label financial news tweets as
  bearish, bullish, or neutral, first with SFT and then with RL against a verifiable
  reward, and measure the gain in macro-F1 over the base model
- **Dataset Suggestions**:
  [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
  - Use `sent_train.csv` for training and `sent_valid.csv` as the held-out set
- **Tasks**:
  - **Preprocess the Tweets**: Turn each tweet into a prompt that asks for one label
    word, keep 2,200 training tweets, and keep the validation file untouched for the
    evaluation
  - **Define the Problem**: Measure the zero-shot macro-F1 of the base model, and fit
    TF-IDF with logistic regression as the classical baseline
  - **Fine-Tune with SFT**: Run `SFTTrainer` on 200 labeled tweets as a warm start, so
    that the model learns the answer format
  - **Optimize with RL**: Run `GRPOTrainer` on the other 2,000 tweets with a reward of
    1 for the correct label and a penalty for extra text, and log the mean reward at
    each step
  - **Evaluate the Models**: Compare the macro-F1, the accuracy, and the rate of
    invalid answers of the base, SFT, RL, and TF-IDF models on the validation tweets
  - **Visualize the Results**: Plot the reward curve, and the confusion matrices of
    the base and RL models
- **Bonus Ideas (Optional)**: Compare the SFT and RL recipe with plain SFT on all the
  training tweets; use a reward that penalizes more the confusion of bullish with
  bearish

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Preprocess the Tweets
  - Result: project dir created and container running with the pinned `trl` version,
    and a table with the number of tweets and the class counts of each split
- Milestone 2: API notebook
  - Project tasks: Fine-Tune with SFT, Optimize with RL
  - Result: `trl.API.ipynb` covering `SFTTrainer`, `RewardTrainer`, `DPOTrainer`,
    `GRPOTrainer`, and a custom reward function
- Milestone 3: Example notebook
  - Project tasks: Define the Problem, Fine-Tune with SFT, Optimize with RL, Evaluate
    the Models, Visualize the Results
  - Result: `trl.example.ipynb` running end to end

## Project 2: Preference Optimization of Cautious Investment Answers

- **Project Objective**: Make a small language model answer investing questions with
  balanced and risk-aware language, using preference pairs and a reward model, and
  measure the drop in overconfident claims
- **Dataset Suggestions**:
  [Finance-Alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca)
  (financial questions and answers, in the file `Cleaned_date.json`)
- **Tasks**:
  - **Load the Questions**: Keep 2,000 questions with an empty `input` that ask for
    advice (e.g., with "should", "invest", or "retirement"), and hold out 200 for the
    evaluation
  - **Build Preference Pairs**: Sample two answers per question from a small instruct
    model such as `Qwen2.5-0.5B-Instruct`, and mark as `chosen` the answer with the
    higher caution score, which adds points for risk language (e.g., "risk",
    "diversif") and subtracts points for overconfident phrases (e.g., "guaranteed",
    "risk-free")
  - **Train a Reward Model**: Fit `RewardTrainer` on the pairs, and report its
    pairwise accuracy on a held-out split
  - **Optimize with DPO**: Run `DPOTrainer` on the pairs, starting from the same small
    instruct model
  - **Evaluate the Answers**: Compare the reward model win rate, the rate of
    overconfident claims measured with a second and different phrase list, and the
    answer length of the base and optimized models on the 200 held-out questions
  - **Analyze the Behavior**: Read ten examples, and look for reward hacking such as a
    boilerplate disclaimer added to every answer
- **Bonus Ideas (Optional)**: Compare `DPOTrainer` with plain `SFTTrainer` on the
  `chosen` answers only

## Project 3: Numeric Reasoning on Financial Reports with GRPO

- **Project Objective**: Train a small language model to answer numeric questions about
  company financial reports with a verifiable reward, and measure the trade-off between
  the accuracy and the drift from the base model
- **Dataset Suggestions**:
  [FinQA](https://github.com/czyssrs/FinQA) (questions over the earnings reports of
  S&P 500 companies, with the gold program and the supporting facts)
  - Use `dataset/train.json` for training and `dataset/dev.json` for the evaluation
- **Tasks**:
  - **Select the Questions**: Keep the questions whose gold `program` has at most two
    operations, and build a short prompt from the supporting facts (`gold_inds`) and
    the question, with 500 questions for training and 200 for the evaluation
  - **Define the Reward**: Parse the final number of the completion, and give 1 if it
    is within 1% of `exe_ans` (accepting percent notation), plus a small bonus for the
    format `Answer: <number>`
  - **Optimize the Policy**: Train a small instruct model with `GRPOTrainer` for three
    values of the KL coefficient `beta`, with three seeds each
  - **Evaluate the Trade-Offs**: Plot the mean reward vs. the KL divergence to the
    reference model, with the standard deviation over the seeds, and report the
    accuracy on the held-out questions for each `beta`
  - **Check the Reasoning Quality**: Measure the share of answers that only copy a
    number of the prompt, and read 20 completions to detect reward hacking
  - **Compare with a Baseline**: Fine-tune with `SFTTrainer` on the best-of-8
    completions by reward, and compare it with the RL models and the base model
- **Bonus Ideas (Optional)**: Repeat the best setting with `PPOTrainer` and compare it
  with `GRPOTrainer`; add a reward for a step-by-step format
