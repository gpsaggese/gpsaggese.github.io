FILE=msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd

- [ ] Review and improve slides

/slides.review $FILE

Make sure it renders correctly:
> gen_slides.py -i $FILE

Implement the restructuring of the slides and fix the high importance issues

- [ ] Add visuals and references to slides

/slides.add_visuals $FILE
/slides.add_references $FILE

Make sure it renders correctly
> gen_slides.py -i $FILE

- [ ] Generate the book chapter

> gen_book_chapter.py -i msml610/01.2 --mode typst_aima --llm_backend hllm_cli_exec --model openrouter/anthropic/claude-opus-4.6 --no_incremental
> run_typst.py --input msml610/book/Lesson01.2-AI_and_Machine_Learning.typ
