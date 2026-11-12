- The cover letter for MBZUAI
  - `cover_letter_mbzuai.md`: text of the letter
  - `paper.tex`: typeset version, with the same text and a similar header as the
    CV
  - `Makefile`: symlink to `../template/Makefile`, like all the paper dirs
  - `gp-saggese-adia.png`: photo in the header

- Build the PDF (`paper.pdf`) from the top of the repo
  ```
  > make -C papers/gp_saggese_coverletter
  ```

- Keep `cover_letter_mbzuai.md` and `paper.tex` in sync when editing the text
