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

// Shrink?
Lesson02.1-A_Map_of_Machine_Learning.smd
Lesson02.2-ML_Paradigms.smd

// Keep
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

# Workflow in short

/slides.lint          01.2  01.3  01.4
/slides.review        01.2  01.3  01.4
/slides.add_visuals   01.2  01.3  .
/slides.add_references 01.2 01.3  .

Not needed
/slides.fix_errors
/slides.reduce_text
/slides.fix_formatting
/slides.add_tutorial_links

> lint_text.py -i     01.2  01.3  .

> gen_slides.py -i msml610/01.3
> grep "^* " msml610/lectures_source/*.smd | wc -l

> gen_book_chapter.py msml610/01.2 --mode typst_aima --llm_backend hllm_cli
gen_book_chapter.py  msml610/01.2 --mode typst_aima --llm_backend hllm_cli -v DEBUG --model claude-opus-4.5
gen_book_chapter.py  msml610/01.2 --mode typst_aima --llm_backend hllm_cli_exec --model openrouter/anthropic/claude-opus-4.6 --no_incremental
gen_book_chapter.py  msml610/00.1 --mode typst_aima --llm_backend hllm_cli_exec --model openrouter/anthropic/claude-opus-4.6 --no_incremental

> /text.humanize       01.2 01.3
> review / edit book chapter

> run_typst.py --input msml610/book/Lesson01.2-AI_and_Machine_Learning.typ

### [x] gen_book_chapter.py

- Print all the actions in a table using the standard function like other scripts
- Make --action render_pdf default

- The entire slide should be sent together and not in chunks as done by
  _process_panel_body and other related functions
  - Make a proposal in 5 bullet points of what to change

### [x] Improve the generation
- Compare gpt-4o-mini to a better model
- hllm_cli vs hllm

- Still lots of tags... modify the prompt to remove them
- Latex not converted correctly
- Do not keep the formatting in the page (e.g., pros vs cons)

- The figures embedded in the text are good, but we need caption and a reference
  in the text

### [ ] Updates

- Use Definition to have the tag on the side

- "References" need to be a larger font

### [ ]
- Keep the LLM chat open so that we don't have to send the same instructions over and
  over (only for the library version)
- Also we can use this to keep track of the old text and make the transitions
  smoother

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
