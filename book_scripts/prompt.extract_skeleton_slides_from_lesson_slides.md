# Goal
- You are a college professor expert in AI, machine learning, and big data
- You need to create the table of content for a chapter of a technical book from
  lecture slides already present


# Step 1
- The relevant variables are
  ```
  DIR="/Users/saggese/src/umd_classes1"
  BOOK="book_springer"
  CHAP="## 08: Causal Data Pipelines"
  ```

- Focus only on the chapter in `<DIR>/<BOOK>/map.md`

- Read the sections in the `<CHAP>` above
  ```
  ### Goals

  ### Topics
  ```

# Step 2
- Extract the TOCs from all the slides `### Lesson Materials` using
  ```
  > extract_toc_from_text.py -i <FILE> --max_level 5 --type header
  ```

# Step 3

## Create TOC slides
- Save the detailed output in the form of headers, slide titles, and where the
  slides come from, e.g.,
  ```
  # Header 1

  ## Header 2
  * <title of slide>
  // From: <lesson_path>:<line num> '<title of slide>'
  * <title of slide>
  // From: data605/lectures_source/Lesson02.3-Data_Pipelines.txt:87 '* <title of slide>'
  * <title of slide>
  // From: msml610/lectures_source/Lesson06.1-Bayesian_Networks.txt:142 '* <title of slide>'
  ...
  ```
  in `<BOOK>/lectures_source/Lesson0<num>.01_<name>.txt`
  - `<lesson_path>` is a repo-root-relative path to the exact source `.txt` /
    `.smd` file, `<line num>` its line, and `'<title of slide>'` that line's
    full text single-quoted verbatim, leading marker included (`* `)
  - If no source slide can be identified, leave a bare `// From:` (nothing
    after the colon) instead of guessing
  - See `.claude/skills/slides.rules.md` ("Source Attribution") for the full
    convention
- The content should be around 30-35 slides

## Update Topics
- Update the topics `### Topics` using the `H1` and `H2` headers from the
  previous section

## Update Goals
- 3 short bullet points of less of 100 characters each explaining what are the
  goals of the chapter
