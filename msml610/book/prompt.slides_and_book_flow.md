# Goal

- Process the slides and the book for the following set of slides
  ```
  SMD_FILE=msml610/lectures_source/Lesson05.1*.smd
  ```

# Workflow

## Create a plan

- Create a plan `plan-msml610/book/prompt.slides_and_book_flow.md` for the following
  actions and keep it updated using [ ], [.], [x] as you go

## Read convention

- Read the conventions for
  - books: `.claude/skills/book_map.rules.md`
  - slides: `.claude/skills/slides.rules.md`
  - typst code: `.claude/skills/typst.rules.md`

## Review and Improve slides

- Run the skill
  ```
  /slides.review <SMD_FILE>
  ```
  - Implement the restructuring of the slides and fix the high importance issues
    reported by the skill

- Make sure the `<SMD_FILE>`>`>`>`>`>`>`>` renders correctly:
  ```
  > gen_slides.py -i <SMD_FILE>
  ```
  - If not fix the problems

- Add visuals and references to the `<SMD_FILE>`
  ```
  /slides.add_visuals <SMD_FILE>
  /slides.add_references <SMD_FILE>
  ```

- Make sure the `<SMD_FILE>` renders correctly:
  ```
  > gen_slides.py -i <SMD_FILE>
  ```
  - If not fix the problems

## Generate the book chapter

- Generate the book chapter for `<SMD_FILE>`
  ```
  > gen_book_chapter.py -i <SMD_FILE> --mode typst_aima --llm_backend hllm_cli_exec --model openrouter/anthropic/claude-opus-4.6 --no_incremental
  ```
  which generates a file `<TYP_FILE>` in `msml610/book/XYZ.typ`

- Humanize
  ```
  /text.humanize <TYP_FILE>
  ```

- Make sure that the generated typst code compiles
  ```
  > run_typst.py --input <TYP_FILE>
  ```
