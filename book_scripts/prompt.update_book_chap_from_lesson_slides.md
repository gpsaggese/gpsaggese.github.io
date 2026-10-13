- Given a file name `<FILE>` (e.g., `Lesson10.2-Causal_Inference_for_Time_Series`)
  - The `<TYPST_FILE>` is `msml610/book/<FILE>.typ` (e.g.,
    `msml610/book/Lesson10.2-Causal_Inference_for_Time_Series.typ`)
  - The corresponding text file `<SLIDE_FILE>` with the original slides is
    `msml610/lectures_source/<FILE>.txt`
    (e.g., `msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.txt`)

- Read the tag `<GIT_HASH>` in the first line of the typst file `<TYPST_FILE>`
  ```
  // git_hash=<GIT_HASH>, timestamp=<TIMESTAMP>
  ```

- Find the changes to the slides `<SLIDE_FILE>` in Git
  ```
  > git diff <TAG> HEAD -- <SLIDE_FILE> >changes.txt

  # E.g.,
  > git diff 382e0cdf HEAD -- msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.txt >changes.txt
  ```

- Update the `<TYPST_FILE>` with the changes `changes.txt` following the
  instructions `book_proposals/prompt.create_typst_book_chap_from_lesson_slides.md` 
