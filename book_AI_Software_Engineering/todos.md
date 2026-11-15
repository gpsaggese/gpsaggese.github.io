# Resources

- book_AI_Software_Engineering/book_map.md

### [ ] Generate Slides

// From book_scripts/README.md
```
claude> /model opus

- Modify book_scripts/prompt.create_lesson_slides_from_chap_toc.md to generate
claude> Execute book_scripts/prompt.create_lesson_slides_from_chap_toc.md without asking for confirmation

> gen_slides.py -i book_AI_Software_Engineering/14.01
> gen_slides.py -i book_AI_Software_Engineering/14.01 --action release

> lint_text.py -i book_AI_Software_Engineering/lectures_source/Lesson05.01*.smd --check

claude> /slides.review Lesson01.01_LLM_Assisted_Code_Generation.smd
claude> /slides.add_visuals Lesson01.01_LLM_Assisted_Code_Generation.smd
claude> /slides.add_references Lesson01.01_LLM_Assisted_Code_Generation.smd
```

```
claude> Execute book_scripts/prompt.slides_and_book_flow.md book_AI_Software_Engineering/lectures_source/Lesson05.01_Skills_Organization_and_Management.smd

> run_typst.py --input msml610/book/Lesson01.2-AI_and_Machine_Learning.typ

> /text.humanize
> /book_chapter.fix_headings
> /book_chapter.improve_text_tags

Edit book chapter

run_typst.py --compress_pdf
> compress_pdf.py --input msml610/book/Lesson01.3*.pdf
```

###
- [ ] Template
  - [ ] book_scripts/prompt.create_lesson_slides_from_chap_toc.md
  - [ ] /slides.review <smd>
  - [ ] /slides.add_visuals <smd>
  - [ ] /slides.add_references <smd>
  - [ ] book_scripts/prompt.slides_and_book_flow.md

- [ ] Lesson01.01_LLM_Assisted_Code_Generation.smd
  - [x] book_scripts/prompt.create_lesson_slides_from_chap_toc.md
  - [ ] /slides.review
  - [.] /slides.add_visuals
  - [x] /slides.add_references
  - [x] book_scripts/prompt.slides_and_book_flow.md

- [ ] Lesson05.01_Skills_Organization_and_Management.smd
  - [ ] book_scripts/prompt.create_lesson_slides_from_chap_toc.md
  - [ ] /slides.review
  - [ ] /slides.add_visuals
  - [ ] /slides.add_references
  - [ ] book_scripts/prompt.slides_and_book_flow.md

- [ ] Lesson11.01_Planning_Software.smd
  - [x] book_scripts/prompt.create_lesson_slides_from_chap_toc.md
  - [ ] /slides.review
  - [ ] /slides.add_visuals
  - [ ] /slides.add_references
  - [ ] book_scripts/prompt.slides_and_book_flow.md

### [ ] Rename chapters

- In book_AI_Software_Engineering/book_map.md
  - Rename / reorg chapters

### [ ] Rename book_AI_Software_Engineering -> book_AI_Sw_Eng

### [.] Merge Into the TOC

- Merge the topics from
  helpers_root/papers/AIgentic_Development_System
  into
  book_AI_Software_Engineering/resources.md

- Merge the resulting 
  helpers_root/papers/AIgentic_Development_System
  into
  book_AI_Software_Engineering/book_map.md

### [ ] Download Books and Add Them to Kindle

- Book_AI_Software_Engineering/book_info.md

### [x] Come up with example
- Read book_AI_Software_Engineering/book_map.md and come up with 3 potential running
  example of a software system that can be used for the entire book
- Output: book_AI_Software_Engineering/running_example.md
- Decision: use `shortlink` (URL shortener service)

### [x] Update with running example
- Add a reference to book_AI_Software_Engineering/running_example.md as running
  example to book_AI_Software_Engineering/book_map.md
- Update Lesson12.01_Planning_Software.smd with book_AI_Software_Engineering/running_example.md
  instead of the pipeline to download and process papers

### [ ] Use example as unit test
- This example should be also used as "unit tests", e.g., running
  prompts to see what is generated

### [ ] Changes to L11.01

- Change "Capturing an Idea" to a slide about describing the idea as an informal
  natural language description

### [.] Review current content

> gen_slides.py -i book_AI_Software_Engineering/00.01

- [x] Lesson00.01_History_of_Software_Engineering.smd
Lesson01.01_LLM_Assisted_Code_Generation.smd
Lesson02.01_AI_Harness.smd
Lesson03.01_AI_Pair_Programming_Workflows.smd
Lesson04.01_Prompt_Engineering_for_Code.smd
Lesson05.01_Skills_Organization_and_Management.smd
Lesson06.01_Human_AI_Collaboration_Patterns.smd
Lesson07.01_Code_Cosmetics.smd
Lesson08.01_Documenting_Code.smd
Lesson11.01_Planning_Software.smd
Lesson12.01_Software_Architecture_Guidelines.smd
Lesson13.01_Principles_of_Good_Software_Design.smd
Lesson14.01_GitHub_Projects_and_Workflows.smd

### [ ] Generate a Typst

From book_scripts/README.md
- book_scripts/prompt.create_typst_book_chap_from_lesson_slides.md
