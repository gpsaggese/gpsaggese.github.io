# Resources

- Book_AI_Software_Engineering/book_map.md

### [ ] Generate Slides

// TODO(ai_gp): Update this table with the files in book_AI_Software_Engineering/lectures_source/ and book_AI_Software_Engineering/book

| Chap                                          | Generate slides | Review slides | Generate book |
| :-------------------------------------------- | :-------------- | :------------ | :------------ |
| **Part I: Foundations of AI-Assisted Coding** |                 |               |               |
| 01, Coding Models                             | yes             |               |               |
| 02, Skills Organization and Management        | yes             |               |               |
| 03, AI Harness                                |                 |               |               |
| 04, AI Pair Programming Workflows             | yes             |               |               |
| 05, Prompt Engineering for Code               | yes             |               |               |
| 06, Human-AI Collaboration Patterns           | yes             |               |               |
| **Part II: Writing and Reviewing Code**       |                 |               |               |
| 07, Code Cosmetics                            |                 |               |               |
| 08, Documenting Code                          |                 |               |               |
| 09, Reviewing Pull Requests                   |                 |               |               |
| 10, Reviewing Existing Code                   |                 |               |               |
| 11, Debugging with AI                         |                 |               |               |
| **Part III: Planning and Architecture**       |                 |               |               |
| 12, Planning                                  |                 |               |               |
| 13, Software Architecture Guidelines          |                 |               |               |
| **Part IV: Delivery and Automation**          |                 |               |               |
| 14, GitHub Projects and Workflows with AI     | yes             |               |               |
| 15, Releasing and Docker with AI              |                 |               |               |
| 16, Setting CI/CD for AI                      |                 |               |               |

### [ ]

// From book_scripts/README.md
```
claude> /model opus

- Modify book_scripts/prompt.create_lesson_slides_from_chap_toc.md to generate the slides
claude> Execute book_scripts/prompt.create_lesson_slides_from_chap_toc.md without asking for confirmation

> gen_slides.py -i book_AI_Software_Engineering/14.01
> gen_slides.py -i book_AI_Software_Engineering/14.01 --action release

> lint_text.py -i book_AI_Software_Engineering/lectures_source/Lesson02.1*.smd --check

claude> Execute msml610/book/prompt.slides_and_book_flow.md book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd

> run_typst.py --input msml610/book/Lesson01.2-AI_and_Machine_Learning.typ

> /text.humanize          .
> /book_chapter.fix_headings      .
> /book_chapter.improve_text_tags .

Edit book chapter

run_typst.py --compress_pdf .
> compress_pdf.py --input msml610/book/Lesson01.3*.pdf
```

### [ ] Rename book_AI_Software_Engineering -> book_AI_Sw_Eng
