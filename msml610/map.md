# Topics

- Each lesson should be around 30 slides and correspond to a chapter of a book
```
> grep "^* " msml610/lectures_source/*.smd | wc -l
```

Lesson | Num slides | Check for error | Add visuals | Slides posted | Generate commentary | Review commentary | Duration | Comm Posted | Tutorial | Video |
Lesson01.1-Class.smd   | 14 | 90% | Yes | Yes | n/a | n/a | n/a | n/a | n/a | n/a |
Lesson01.2-AI_and_Machine_Learning.smd | 14 
Lesson01.3-The_Foundations_of_AI.smd (13)
Lesson01.4-Brief_History_of_AI.smd (26)

Lesson02.1-A_Map_of_Machine_Learning.smd
Lesson02.2-ML_Paradigms.smd

Lesson02.3-ML_Techniques_Input_Processing.smd
Lesson02.4-ML_Techniques_Model_Learning.smd
Lesson02.5-ML_Techniques_Model_Evaluation.smd
Lesson02.6-ML_Techniques_How_To_Do_Research.smd

Lesson03.1-Knowledge_representation.smd

// Move
Lesson04.1-Models.smd
Lesson04.2-Models.smd
Lesson04.3-Models.smd

## Learning Theory
Lesson05.1-Learning_Theory.smd
Lesson05.2-Overfitting.smd
Lesson05.3-Learn_Validation.smd

## Probabilistic ML
Lesson06.1-Bayesian_Networks.smd
Lesson06.2-Using_Bayesian_Networks.smd
Lesson07.1-Intro_to_Probabilistic_Programming.smd
Lesson07.2-Posterior_Based_Decisions.smd
Lesson07.3-Hierarchical_Models.smd
Lesson07.4-Generalized_Linear_Models.smd
Lesson07.5-Bayesian_Model_Comparison.smd

## Causal ML
Lesson08.1-Causal_AI_intro.smd
Lesson08.2-Causal_Networks.smd
Lesson08.3-Do_Calculus.smd

Lesson08.4.smd
Lesson08.5-Experimentation.smd

## Forecasting and Decision Making
Lesson09.1-Reasoning_over_time.smd
Lesson09.2-Hidden_Markov_Models.smd
Lesson09.3-Multi_Armed_Bandits.smd
Lesson09.7-Advanced_Bandits.smd
Lesson09.4-gh_Filter.smd
Lesson09.5-Kalman_Filter.smd
Lesson09.6-Dynamic_Bayesian_Networks.smd

Lesson10.1-Timeseries_forecasting.smd
Lesson10.2-Causal_Inference_for_Time_Series.smd
Lesson11.1-Decision_Making_with_Causal_Models.smd

// Move
Lesson11.2-Probabilistic_deep_learning.smd

// Move
Lesson12.1-Reinforcement_learning.smd

// Move
Lesson12.2-Causal_Discovery.smd

// ?
Lesson13.1-Explainability.smd

> count_lecture_slides.py msml610
WARNING: Can't find s3fs: continuing
15:58:02 - INFO  hdbg.py init_logger:1167                               Saving log to file '/Users/saggese/src/umd_classes1/class_scripts/count_lecture_slides.py.log'
15:58:02 - INFO  hdbg.py init_logger:1177                               > /Users/saggese/src/umd_classes1/class_scripts/count_lecture_slides.py msml610
15:58:02 - INFO  count_lecture_slides.py _main:178                      DIR=msml610, FORMAT=markdown
15:58:02 - INFO  count_lecture_slides.py _collect_stats:100             Scanning directory: msml610/lectures_source
| File                                                 |   Slides |   H1 |   H2 |   H3 |   Lines |   Words |   Chars |
|------------------------------------------------------|----------|------|------|------|---------|---------|---------|
| Lesson00.1-Test.smd                                  |        2 |    1 |    1 |    0 |      63 |     260 |    2034 |
| Lesson01.1-Intro.smd                                 |       14 |    3 |    0 |    0 |     115 |     362 |    3797 |
| Lesson01.1.aux.md                                    |        0 |    6 |    0 |    0 |     890 |    2595 |   29842 |
| Lesson01.2-AI_and_Machine_Learning.smd               |       19 |    2 |    0 |    0 |     516 |    2165 |   16315 |
| Lesson01.3-The_Foundations_of_AI.smd                 |       16 |    1 |    7 |    0 |     559 |    1837 |   17613 |
| Lesson01.4-Brief_History_of_AI.smd                   |       28 |    2 |    3 |    0 |     989 |    3406 |   29721 |

| Lesson02.1-A_Map_of_Machine_Learning.smd             |        7 |    1 |    0 |    0 |     258 |    1007 |   11710 |
| Lesson02.2-ML_Paradigms.smd                          |       19 |    1 |    4 |    0 |     441 |    2231 |   15799 |
| Lesson02.3-ML_Techniques_Input_Processing.smd        |        9 |    0 |    1 |    0 |     171 |     707 |    5061 |
| Lesson02.4-ML_Techniques_Model_Learning.smd          |       17 |    0 |    1 |    2 |     432 |    1902 |   14077 |
| Lesson02.5-ML_Techniques_Model_Evaluation.smd        |       38 |    0 |    3 |    4 |     771 |    3948 |   27072 |
| Lesson02.6-ML_Techniques_How_To_Do_Research.smd      |       17 |    0 |    1 |    2 |     372 |    1819 |   12283 |

| Lesson03.1-Knowledge_representation.smd              |       34 |    1 |    5 |    0 |     962 |    4548 |   32463 |
| Lesson03.2-Propositional_and_first_order_logic.smd   |       27 |    2 |    4 |    0 |     555 |    2966 |   19394 |
| Lesson03.3-Non_classical_logics.smd                  |       26 |    3 |    2 |    1 |     623 |    2449 |   19315 |
| Lesson04.1-Models.smd                                |       37 |    0 |    5 |    0 |     793 |    3512 |   23802 |
| Lesson04.2-Models.smd                                |       54 |    0 |    4 |    0 |    1051 |    4860 |   31538 |
| Lesson04.3-Models.smd                                |       35 |    0 |    3 |    0 |     639 |    2960 |   20151 |
| Lesson05.1-Learning_Theory.smd                       |       27 |    3 |    0 |    0 |     992 |    4738 |   31665 |
| Lesson05.2-Overfitting.smd                           |       28 |    3 |    0 |    0 |     811 |    3243 |   22212 |
| Lesson05.3-Learn_Validation.smd                      |       18 |    1 |    2 |    0 |     438 |    1773 |   12998 |
| Lesson06.1-Bayesian_Networks.smd                     |       27 |    2 |    6 |   12 |     872 |    3558 |   27037 |
| Lesson06.2-Using_Bayesian_Networks.smd               |       37 |    4 |    0 |    0 |    1248 |    4833 |   38148 |
| Lesson07.1-Intro_to_Probabilistic_Programming.smd    |       26 |    2 |    3 |    0 |     638 |    2560 |   19283 |
| Lesson07.2-Posterior_Based_Decisions.smd             |       27 |    1 |    3 |    1 |     500 |    2034 |   15229 |
| Lesson07.3-Hierarchical_Models.smd                   |       11 |    1 |    0 |    0 |     171 |     674 |    5229 |
| Lesson07.4-Generalized_Linear_Models.smd             |       29 |    1 |    3 |    0 |     620 |    2137 |   17476 |
| Lesson07.5-Bayesian_Model_Comparison.smd             |       32 |    1 |    5 |    4 |     974 |    5624 |   36464 |
| Lesson08.1-Causal_AI_intro.smd                       |       22 |    2 |    3 |    0 |     424 |    2134 |   15694 |
| Lesson08.2-Causal_AI_concepts.smd                    |       10 |    1 |    3 |    0 |     209 |     864 |    6329 |
| Lesson08.3-Causal_AI_in_business.smd                 |       21 |    2 |    2 |    0 |     426 |    2017 |   15825 |
| Lesson08.4-Causal_networks.smd                       |       27 |    1 |    5 |    1 |    1368 |    4214 |   37534 |
| Lesson08.5-Do_calculus.smd                           |       23 |    5 |    0 |    0 |     821 |    3057 |   23641 |
| Lesson08.6-Causal_inference_intro.smd                |       47 |    3 |    0 |    0 |    1230 |    5359 |   38314 |
| Lesson08.7-Causal_experiments.smd                    |       34 |    5 |    8 |    0 |     602 |    2875 |   20298 |
| Lesson08.8.Causal_Linear_Regression.smd              |       26 |    2 |    0 |    0 |     619 |    3213 |   23336 |
| Lesson08.9-Effect_heterogeneity_and_Metalearners.smd |       22 |    2 |    0 |    0 |     514 |    2945 |   20614 |
| Lesson08.X-Causal_inference.smd                      |        5 |    5 |    2 |    0 |     178 |     808 |    5641 |
| Lesson09.1-Reasoning_over_time.smd                   |       33 |    1 |    3 |    0 |     877 |    3936 |   29168 |
| Lesson09.2-Hidden_Markov_Models.smd                  |       19 |    4 |    1 |    0 |     457 |    2254 |   16768 |
| Lesson09.3-Multi_Armed_Bandits.smd                   |       36 |    6 |    0 |    0 |     971 |    5330 |   40568 |
| Lesson09.4-gh_Filter.smd                             |       11 |    1 |    0 |    0 |     326 |    1320 |    9956 |
| Lesson09.5-Kalman_Filter.smd                         |       42 |    3 |    2 |    0 |     819 |    3981 |   27335 |
| Lesson09.6-Dynamic_Bayesian_Networks.smd             |       12 |    1 |    0 |    0 |     286 |    1236 |    9696 |
| Lesson09.7-Advanced_Bandits.smd                      |       28 |    7 |    0 |    0 |     719 |    4106 |   28588 |
| Lesson09.8-Classical_search_algorithms.smd           |       55 |    4 |    0 |    0 |    2085 |   10422 |   75509 |
| Lesson09.9-MonteCarlo_Tree_Search.smd                |       36 |    5 |    0 |    0 |     874 |    4821 |   33863 |
| Lesson10.1-Timeseries_forecasting.smd                |       65 |    4 |    8 |    0 |    1425 |    7656 |   54964 |
| Lesson10.2-Causal_Inference_for_Time_Series.smd      |       37 |    6 |    6 |    0 |     831 |    4117 |   29687 |
| Lesson11.1-Decision_Making_with_Causal_Models.smd    |       53 |    5 |   13 |    0 |    1282 |    6480 |   45617 |
| Lesson11.2-Probabilistic_deep_learning.smd           |      104 |   12 |    6 |    0 |    1991 |    9950 |   71167 |
| Lesson12.1-Reinforcement_learning.smd                |       64 |    2 |    9 |    0 |    2188 |    9570 |   73546 |
| Lesson12.2-Causal_Discovery.smd                      |       30 |    4 |    5 |    0 |     661 |    3228 |   24052 |
| Lesson13.1-Explainability.smd                        |       34 |    3 |    3 |    0 |     866 |    4096 |   30268 |

# Workflow in short

/slides.review            03.1   .    .   .   .   .

/slides.add_visuals       .
/slides.add_references    .
/slides.lint              .
/slides.fix_rendered_pdf  .

Edit slides               03.1

Not needed
/slides.fix_errors
/slides.reduce_text
/slides.fix_formatting
/slides.add_tutorial_links

lint_text.py              .
> lint_text.py -i msml610/lectures_source/Lesson02.1*.smd

gen_slides.py             .
> gen_slides.py -i msml610/01.3
> grep "^* " msml610/lectures_source/*.smd | wc -l

tutorials

gen_book_chapter          .
gen_book_chapter.py -i msml610/01.2 --mode typst_aima --llm_backend hllm_cli_exec --model openrouter/anthropic/claude-opus-4.6 --no_incremental

run_typst.py              .
> run_typst.py --input msml610/book/Lesson01.2-AI_and_Machine_Learning.typ

> /text.humanize          .

Edit book chapter

run_typst.py --compress_pdf .
> compress_pdf.py --input msml610/book/Lesson01.3*.pdf

### [ ] Add cc loop in gen_book_chapters
- Instead of using an LLM use cc agent
  - Iterate until it compiles
  - run_typst.py --input msml610/book/Lesson01.4-Brief_History_of_AI.typ --output msml610/book/Lesson01.4-Brief_History_of_AI.pdf --action render_images --skip_action open_pdf

- Keep the LLM chat open so that we don't have to send the same instructions over and
  over (only for the library version)
- Also we can use this to keep track of the old text and make the transitions
  smoother

### [x] gen_book_chapter.py

- Print all the actions in a table using the standard function like other scripts
- Make --action render_pdf default

- The entire slide should be sent together and not in chunks as done by
  _process_panel_body and other related functions
  - Make a proposal in 5 bullet points of what to change

### [x] Improve the generation
- [x] Compare gpt-4o-mini to a better model
  - hllm_cli vs hllm

- [x] Still lots of tags... modify the prompt to remove them
  - Latex not converted correctly
  - Do not keep the formatting in the page (e.g., pros vs cons)

- [x] The figures embedded in the text are good, but we need caption and a reference
  in the text

- [ ] Use Definition to have the tag on the side
- x ] "References" need to be a larger font

# Workflows

## Overview

- Extract headers and create a comprehensive syllabus from all lecture materials
  using the `for_loop_lessons.py` orchestration script

## Slides

### Iterate on the Slides

- Generate slides when editing the source
  ```bash
  > gen_slides.py -i msml610/lectures_source/Lesson01.1-Intro.smd
  > gen_slides.py -i msml610/01.1 --daemon
  > gen_slides.py -i msml610/01.1 --daemon
  ```

- The file is generated in `lectures_pdf.tmp`

## Check Slides

```
claude> /slides.criticize msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd
```

## Slides Commentary

### Generate for One Lecture
- Generate one lecture
  ```
  > gen_lecture_commentary.py msml610/01.1 --image_type jpg
  ```

### Generate for All Lectures

- Generate all the lectures
  ```
  > for_loop_lessons.py --class data605 --action generate_lecture_commentary --lectures "01.1-02"

  # Check out.
  > publish_class_links.py --dir msml610 --out_file ./links.html --do_not_fail_on_warnings --use_master
  > open book_springer/lecture_commentary/Lesson04.1_Knowledge_Representation.book_chapter.html
  ```

### Publish the lecture commentary on the website

website/update_class_links.sh

## Update Slides Commentary

```
claude> Execute class_scripts/prompt.update_lecture_commentary.md on msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd
```

## Course Syllabus

### Generate Complete Course Syllabus

- Extract all lecture headers and create a consolidated syllabus:

  ```bash
  > cd /Users/saggese/src/umd_classes1
  > for_loop_lessons.py --class msml610 --action generate_toc
  ```

This generates:
- **Output file**: `msml610/all_tocs.md`
- **Content**: All lecture headers organized hierarchically (up to 5 levels deep)
- **Format**: Markdown with lecture structure preserved

### Generate Syllabus for Specific Lectures

- Extract headers from a subset of lectures using pattern matching:
  ```bash
  # Single lecture pattern
  > for_loop_lessons.py --class msml610 --lectures "01*" --action generate_toc

  # Multiple lecture patterns (colon-separated)
  > for_loop_lessons.py --class msml610 --lectures "01*:02*:03.1" --action generate_toc

  # Continuous range (inclusive)
  > for_loop_lessons.py --class msml610 --lectures "01.1-03.2" --action generate_toc
  ```

### Output Format

- The syllabus markdown file contains structured headers with proper indentation:
  ```markdown
  # Lesson01.1-Intro.smd

  ## Main Topic
  ### Subtopic 1
  #### Sub-subtopic
  ### Subtopic 2

  # Lesson01.2-Topic.smd

  ## Another Main Topic
  ...
  ```

- This provides a complete overview of the course curriculum and lecture structure
