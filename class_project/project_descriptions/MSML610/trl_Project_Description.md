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
- Create `tutorials/trl/`, since it does not exist yet
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

## Project 1: Style Optimization of Text Generation

- **Difficulty**: 1 (Easy)
- **Project Objective**: Steer GPT-2 towards a Shakespearean style by optimizing a
  reward, and measure the gain over the base model
- **Dataset Suggestions**:
  [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
- **Tasks**:
  - **Preprocess the Text**: Split the text into prompts of 16 words, with the
    following 30 words as the reference, and hold out 10% for the evaluation
  - **Define the Problem**: Write a reward function that is the fraction of words of
    the Shakespeare vocabulary minus a repetition penalty, and measure the reward of
    the base GPT-2
  - **Fine-Tune with SFT**: Run `SFTTrainer` on the training text as the supervised
    starting point
  - **Optimize with RL**: Run `GRPOTrainer` with the reward function, and log the
    mean reward at each step
  - **Evaluate the Model**: Compare the mean reward, the perplexity, and the
    repetition rate of the base, SFT, and RL models on the held-out prompts
  - **Visualize the Results**: Plot the reward curve, and show five prompts with the
    generation of each model
- **Bonus Ideas (Optional)**: Compare two different reward functions; test the effect
  of the KL coefficient

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Preprocess the Text
  - Result: `tutorials/trl/` container running with the pinned `trl` version, and a
    table with the number of prompts and references
- Milestone 2: API notebook
  - Project tasks: Fine-Tune with SFT, Optimize with RL
  - Result: `trl.API.ipynb` covering `SFTTrainer`, `RewardTrainer`, `DPOTrainer`,
    `GRPOTrainer`, and a custom reward function
- Milestone 3: Example notebook
  - Project tasks: Define the Problem, Fine-Tune with SFT, Optimize with RL, Evaluate
    the Model, Visualize the Results
  - Result: `trl.example.ipynb` running end to end

## Project 2: Preference Optimization of a Dialogue Model

- **Difficulty**: 2 (Medium)
- **Project Objective**: Make a small dialogue model answer in a more polite and
  positive way, using preference pairs and a reward model
- **Dataset Suggestions**:
  [DailyDialog](https://huggingface.co/datasets/li2017dailydialog/daily_dialog)
- **Tasks**:
  - **Load the Dialogues**: Build the (context, response) pairs from DailyDialog,
    using the last utterance of the context as the prompt
  - **Build Preference Pairs**: Sample two responses per context from
    `DialoGPT-small`, and mark as `chosen` the one with the higher positive sentiment
    score
  - **Train a Reward Model**: Fit `RewardTrainer` on the pairs, and report its
    pairwise accuracy on a held-out split
  - **Optimize the Dialogue Model**: Run `DPOTrainer` on the pairs, starting from
    `DialoGPT-small`
  - **Evaluate the Responses**: Compare the reward model win rate, the mean
    sentiment, and the response length of the base and optimized models on 200
    held-out contexts
  - **Analyze the Behavior**: Read ten examples, and look for reward hacking such as
    generic or repeated replies
- **Bonus Ideas (Optional)**: Compare `DPOTrainer` with plain `SFTTrainer` on the
  `chosen` responses only

## Project 3: Reward and KL Trade-Offs in Customer Replies

- **Difficulty**: 3 (Hard)
- **Project Objective**: Train a model to write positive replies to unhappy airline
  passengers, and measure the trade-off between the reward and the drift from the
  base model
- **Dataset Suggestions**:
  [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- **Tasks**:
  - **Select the Prompts**: Keep the negative tweets, remove the mentions and the
    links, and use them as prompts
  - **Define the Reward**: Combine the positive score of a pre-trained sentiment
    classifier on the reply with a penalty for replies over 40 tokens
  - **Optimize the Policy**: Train GPT-2 with `GRPOTrainer` for three values of the
    KL coefficient `beta`, with three seeds each
  - **Evaluate the Trade-Offs**: Plot the mean reward vs. the KL divergence to the
    reference model, with the standard deviation over the seeds
  - **Check the Text Quality**: Measure the distinct-2 ratio of the replies, and read
    20 samples to detect reward hacking
  - **Compare with a Baseline**: Fine-tune with `SFTTrainer` on the best-of-8 replies
    by reward, and compare it with the RL model
- **Bonus Ideas (Optional)**: Add a second reward for the relevance of the reply to
  the tweet, and study how the two rewards trade off
