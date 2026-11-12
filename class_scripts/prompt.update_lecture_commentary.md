---
description: Update the content of slide commentary given the changes in the corresponding slides
model: opus
---

- The user will pass you an `.smd` file with slide content `<SOURCE>`
  - E.g., msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd

- You will find the corresponding <TARGET>
  - E.g., msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.book_chapter.md
  - If you can't find it, report an error

- Find and incorporate the changes from `<SOURCE>` into `<TARGET>`

- The file `<TARGET>` contains a header with the last version, in terms of Git hash
  and timestamp, of the material used to generate the current version of
  `<TARGET>`
  - E.g.,
    ```text
    <!-- git_hash=<GIT_HASH> timestamp=<TIMESTAMP> -->
    ```
  - E.g.,
    ```text
    <!-- git_hash=083e7694c-l9l timestamp=20260827_123953 -->
    ```

- Find what changed in `<SOURCE>` from `<GIT_HASH>` to now
  ```
  > git diff <HASH> -- <SOURCE> >changes.txt
  ```
, and modify `<TARGET>` to incorporate those changes

- Follow the same style as `<TARGET>`
  - Follow `class_scripts/prompt.generate_lecture_commentary.md` to understand the
    style of the content
