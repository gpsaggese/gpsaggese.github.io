- The format of a book map is described here
  `.claude/skills/book_map.rules.md`

# All prompts

```
> ls -1 book_scripts/*prompt*
book_scripts/prompt.create_lesson_slides_from_chap_toc.md
book_scripts/prompt.create_lesson_slides_from_topic.md
book_scripts/prompt.create_latex_book_chap_from_lesson_slides.md
book_scripts/prompt.create_typst_book_chap_from_lesson_slides.md
book_scripts/prompt.expand_skeleton_slides.md
book_scripts/prompt.extract_skeleton_slides_from_lesson_slides.md
book_scripts/prompt.find_references_to_chap_toc.md
book_scripts/prompt.inline_lesson_slides_from_skeleton_slides.md
book_scripts/prompt.update_book_chap_from_lesson_slides.md
```

# Prompts

## Create `lesson_slides` from a chapter toc
- E.g., for the agentic AI book, there is a `chap_toc` and we need to create
  `lesson_slides`
- Reads the chapter TOC, `### Topics`, and `### Lesson Materials` from
  `book_map.md`
- Creates a skeleton of titles, asks the user for approval, then calls
  `slides.write` for 30-35 slides and saves a new
  `lectures_source/Lesson<CHAP_NUM>.<ID>_<Title>.smd`
- `book_scripts/prompt.create_lesson_slides_from_chap_toc.md`

## Create `lesson_slides` from a topic list (no book map yet)
- Same goal and workflow as the prompt above, but topics are given as an ad
  hoc list inline in the prompt instead of being read from a chapter's
  `### Topics` in `book_map.md`
- Useful when the book map does not have that chapter yet, or for a course
  such as `msml610`
- `book_scripts/prompt.create_lesson_slides_from_topic.md`

## Find material
- Complete the pointers and add references to `resources.md`
  `book_scripts/prompt.find_references_to_chap_toc.md`

## Create `skeleton_slides` from `book_map` and `slides_tutorials`
- In practice we read the TOC for a chapter in the `book_map.md` and find the
  material already available for it
- `book_scripts/prompt.extract_skeleton_slides_from_lesson_slides.md`

## Inline slides in a skeleton slide
- `book_scripts/prompt.inline_lesson_slides_from_skeleton_slides.md`
- Mechanically copies existing slide content into a skeleton, following each
  slide's `// From: <file>:<line num> '<title>'` pointer
- This is also implemented as Python script since it's a mechanical operation
  ```
  > class_scripts/inline_content_in_skeleton_slides.py
  > vi $(grep -riL "Score:" *.md | while IFS= read -r f; do stat -f "%m%t%N" "$f"; done | sort -n | head -10 | cut -f2- | tr '\n' '\0' | xargs -0)
  ```
- E.g.,
  ```
  claude> Execute /Users/saggese/src/notes1/book_scripts/prompt.inline_lesson_slides_from_skeleton_slides.md on book_springer/lectures_source/Lesson05.1_Probabilistic_ML.txt
  ```

- Check that the result is correct with:
  ```
  > gen_slides.py book_springer/05.1
  ```

## Expand a skeleton whose slides point to background files
- Same shape as "Inline slides in a skeleton slide" above, but each skeleton
  slide's `// From: <file>` pointer names a whole background file (e.g. a
  rules doc) rather than a specific source slide/line, so the content must be
  generated from that background reading instead of copied verbatim
- `book_scripts/prompt.expand_skeleton_slides.md`

// TODO(gp): Likely a near-duplicate of "Inline slides in a skeleton slide"
// above (same skeleton -> full-slides direction, different `// From:`
// semantics); consider merging as a documented variant of that prompt.

## Generate lectures commentary
- To generate a commentary in `$DIR/lectures_commentary`
  ```
  > gen_lecture_commentary.py book_springer/04.1 --image_type jpg

  > for_loop_lessons.py --class data605 --action generate_lecture_commentary --lectures "01.1-02"

  > publish_class_links.py --dir book_springer --out_file ./links.html --do_not_fail_on_warnings --use_master

  > open book_springer/lecture_commentary/Lesson04.1_Knowledge_Representation.book_chapter.html
  ```

- Check
  ```
  > ls -1 book_springer/lectures_commentary/Lesson*.{pdf,md,html}
  book_springer/lectures_commentary/Lesson02.1_From_Data_Science_To_Decision_Science.book_chapter.html
  book_springer/lectures_commentary/Lesson02.1_From_Data_Science_To_Decision_Science.book_chapter.md
  book_springer/lectures_commentary/Lesson02.1_From_Data_Science_To_Decision_Science.book_chapter.pdf
  book_springer/lectures_commentary/Lesson04.1_Knowledge_Representation.book_chapter.md
  book_springer/lectures_commentary/Lesson05.1_Probabilistic_ML.book_chapter.md
  ```

## Convert slides to typst or Latex text for a book chapter
  `book_scripts/prompt.create_typst_book_chap_from_lesson_slides.md`
  `book_scripts/prompt.create_latex_book_chap_from_lesson_slides.md`

- Execute
  ```
  claude> Execute book_scripts/prompt.create_latex_book_chap_from_lesson_slides.md on 
  ```

  ```
  > ls -1 book_springer/book/Lesson*.tex
  book_springer/book/Lesson02.01_From_Data_Science_To_Decision_Science.tex
  book_springer/book/Lesson02.02_Integrating_Causality_And_Probability_in_ML.tex
  book_springer/book/Lesson04.01_Knowledge_Representation.tex
  ```

## Update the a Typst book chapter when the corresponding `lesson_slides` are updated
- `book_scripts/prompt.update_book_chap_from_lesson_slides.md`
- `book_scripts/prompt.springer.add_references.md`

// TODO(gp): This prompt currently only follows the Typst variant of the
// "Convert slides to typst or Latex" prompt above; update it to branch on
// the chapter file's extension (`.typ` -> typst prompt, `.tex` -> latex
// prompt) now that a LaTeX variant also exists.
