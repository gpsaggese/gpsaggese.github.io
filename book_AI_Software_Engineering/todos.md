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

> lint_text.py -i book_AI_Software_Engineering/lectures_source/Lesson02.1*.smd --check

claude> /slides.review Lesson01.01_LLM_Assisted_Code_Generation.smd
claude> /slides.add_visuals Lesson01.01_LLM_Assisted_Code_Generation.smd
claude> /slides.add_references Lesson01.01_LLM_Assisted_Code_Generation.smd
```

```
claude> Execute msml610/book/prompt.slides_and_book_flow.md book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd

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
  - [ ] msml610/book/prompt.slides_and_book_flow.md

- [ ] Lesson01.01_LLM_Assisted_Code_Generation.smd
  - [x] book_scripts/prompt.create_lesson_slides_from_chap_toc.md
  - [ ] /slides.review
  - [.] /slides.add_visuals
  - [x] /slides.add_references
  - [x] msml610/book/prompt.slides_and_book_flow.md

- [ ] Lesson02.01_Skills_Organization_and_Management.smd
  - [ ] book_scripts/prompt.create_lesson_slides_from_chap_toc.md
  - [ ] /slides.review
  - [ ] /slides.add_visuals
  - [ ] /slides.add_references
  - [ ] msml610/book/prompt.slides_and_book_flow.md

- [ ] Lesson12.01_Planning_Software.smd
  - [x] book_scripts/prompt.create_lesson_slides_from_chap_toc.md
  - [ ] /slides.review
  - [ ] /slides.add_visuals
  - [ ] /slides.add_references
  - [ ] msml610/book/prompt.slides_and_book_flow.md

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

### [ ] Come up with a running example of a sw system
- That can be used end-to-end
