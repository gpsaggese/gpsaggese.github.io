# Prompt Fine-Tuning via an Automated Input/Output Loop

## Status

- **Status:**: draft
- **Complete Specs:**: 30%
- **Assignee:**: TBD

## Core Idea

- Prompts are written by hand and tuned by feel, with no training set, no
  held-out set, and no stopping criterion
- Proposal: treat prompt writing as a supervised fitting problem
  - The dataset is a directory of inputs paired with desired outputs
  - The parameters are the text of the prompt
  - The loss is a similarity between generated and target output
  - The optimizer is an LLM that reads the diff and edits the prompt
- Convert the manual loop below into a Python script, so the iteration is
  deterministic, logged, and reproducible:
  ```text
  Your goal is to create a prompt that summarizes text in the same way I would do
  it

  # Step 1
  - Read `<TEXT>` https://rlhfbook.com/c/01-introduction and save it into `input.md`

  # Step 2
  - Loop
    - Execute `prompt.md` to generate a summary of the <TEXT> in file `output.md`
    - Compare the content of `output.md` to the desired `target.md`
    - Modify `prompt.md` in order to generate a summary of `<TEXT>` closer to
      `target.md`
  - Keep iterating loop until you generate you generate an output.md from <TEXT>

  # Step 3
  - Evaluate out of sample
  ```
- The non-obvious risk is the one that makes this a research question rather
  than a scripting task: with a handful of examples and an expressive
  optimizer, the prompt overfits, i.e., it memorizes the training targets
  instead of capturing the style. The out-of-sample step in Step 3 is the
  whole experiment, not an afterthought

## Formalization

- Dataset of pairs, split into train and held-out:
  ```text
  D = {(x_1, y_1), ..., (x_n, y_n)}
  ```
- Prompt optimization objective:
  ```text
  p* = argmax_p (1/|D_train|) * sum_i sim(f(p, x_i), y_i)
  ```
  - $f(p, x)$ is the model output under prompt $p$ on input $x$
  - $sim$ is the scoring function
- The optimizer step is textual rather than numeric:
  ```text
  p_{t+1} = LLM_edit(p_t, {(x_i, f(p_t, x_i), y_i)}_i, critique_t)
  ```
- Generalization gap is the quantity of interest:
  ```text
  gap = score(p*, D_train) - score(p*, D_heldout)
  ```
- Effective capacity grows with the number of candidate prompts evaluated, so
  the same selection penalty as in `draft.Backtesting_Complexity.md` applies:
  ```text
  penalty ~ sqrt(log(N_prompts_tried) / n_heldout)
  ```
- Stopping rule: stop when held-out score stops improving, not when training
  score saturates

## Key Examples

- **Summarization style transfer**: 10 documents with hand-written target
  summaries, and the loop learns the length, ordering, and level of detail
  the author uses
- **Rule-application prompt**: inputs are files with known violations and
  targets are the corrected files, so $sim$ can be an exact diff rather than
  a fuzzy score
- **Overfitting case**: the optimizer inserts the literal content of a
  training target into the prompt, which drives training score to 1 and
  held-out score down, so prompt length must be constrained
- **Failure mode**: $sim$ is an LLM judge with the same bias as the generator,
  so the loop optimizes the judge rather than the task
- **Failure mode**: targets are inconsistent with each other (the author wrote
  them at different times), so no prompt can fit them and the loop oscillates

## Questions

1. How many training pairs are needed before the learned prompt generalizes,
   and does the requirement scale with task complexity?
2. Which scoring function works best: embedding similarity, an LLM judge, or
   a task-specific deterministic metric? Does the choice change the optimum
   or only the convergence speed?
3. Should the optimizer be constrained (fixed prompt length budget, no
   verbatim copying from targets) to prevent memorization?
4. Is the resulting prompt transferable across models, or does it encode
   quirks of the model it was tuned against?
5. If prompts can be fit from examples, prompt engineering becomes a data
   collection problem, and the scarce resource is labeled targets rather than
   prompt-writing craft

## Research Topics

- **Existing automatic prompt optimizers**: APE, OPRO, TextGrad, and DSPy
  optimizers, as baselines the hand-rolled loop must beat or justify
- **Scoring functions**: deterministic metrics vs LLM judges, and judge
  agreement with human preference
- **Regularization**: prompt length caps, verbatim-copy detection, and
  cross-validation over the example set
- **Experiment logging**: recording every candidate prompt and its scores so
  the number of trials is known and the selection penalty is computable
- **Repo integration**: the harness should read the inputs and targets from a
  directory layout and emit the tuned prompt as a skill file

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: build the harness
  - Define the directory layout for inputs, targets, and prompt versions
  - Implement the generate, score, critique, and edit loop with full logging
    of every candidate prompt and score
  - This is the result: a script that reproduces the manual loop end to end
    on one task

- Milestone 2: add the held-out protocol
  - Split examples into train and held-out, and report both scores per
    iteration
  - Implement the early-stopping rule on held-out score
  - This is the result: learning curves showing the generalization gap

- Milestone 3: compare scoring functions and regularizers
  - Run the same task with an embedding metric, an LLM judge, and a
    deterministic metric
  - Add the prompt-length cap and verbatim-copy check
  - This is the result: a recommendation on scoring and regularization, with
    measured effect on the gap

- Milestone 4: benchmark against existing optimizers
  - Run DSPy or OPRO-style optimization on the same dataset
  - Test transfer of the tuned prompt to a second model
  - This is the result: evidence on whether the custom loop is competitive
    and whether tuned prompts transfer

## References

- Zhou et al., _Large Language Models Are Human-Level Prompt Engineers (APE)_.
  (2023)
- Yang et al., _Large Language Models as Optimizers (OPRO)_. (2024)
- Yuksekgonul et al., _TextGrad: Automatic Differentiation via Text_. (2024)
- Khattab et al., _DSPy: Compiling Declarative Language Model Calls into
  Self-Improving Pipelines_. (2024)
