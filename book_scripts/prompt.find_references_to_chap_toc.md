# Goal
- Find and add references to authoritative and relevant sources for a chapter of
  an advanced book about the given topic (in the form of books, journal papers,
  and local slides)

# Workload
- Information about the entire book flow is at `book_proposals/README.md`

- The relevant variables are:
  ```
  DIR="/Users/saggese/src/umd_classes1"
  BOOK="book_AI_Software_Engineering"
  CHAP="## 02: AI Pair Programming Workflows"
  ```

- Read carefully the entire table of content `<DIR>/<BOOK>/map.md` of the book

## Step 1: Find content
- Your task is to review the content of a chapter of a book in `<CHAP>` from
  `<DIR>/<BOOK>/map.md` and add references to books and technical papers that
  apply to the specific chapter

- Find related academic references (papers, journals, conference papers,
  books, or authoritative articles) about the requested topics
  - Prefer sources from Google Scholar, arXiv, IEEE, ACM, Springer, Elsevier,
    official documentation, and major tech research blogs
  - Add direct arXiv / free-access versions where available
  - Include working URLs for each reference when possible

- Prefer recent references rather than old

## Step 2: Update the map.md file
- Add or improve sections `### Local Material`, `### Books`, `### Papers`
  `### Internet resources` in `<DIR>/<BOOK>/map.md` with the newly found content
  - Local materials are on the local disk (their location is in sections 
    like `# General Resources` in `<DIR>/<BOOK>/map.md`, in files like
    `<DIR>/<BOOK>/resources.md`)
- E.g.,

  ```
  ### Local Material
  - `msml610/lectures_source/Lesson08.1-Causal_AI_intro.txt`
  - `msml610/lectures_source/Lesson08.5-Experimentation.txt`
  
  ### Books
  - Brynjolfsson et al, "Machine, Platform, Crowd", 2021
  - Angrist et al, "Probability, Statistics, and Causal Inference", 2020
  
  ### Papers
  - Imbens et al., "Recent Developments in the Econometrics of Program
    Evaluation", 2018 https://www.nber.org/papers/w24318
  - Kaddour et al., "Challenges and Opportunities with Causal Discovery
    Algorithms", 2023 https://www.nature.com/articles/s41598-020-59669-x
  ```

- For books and papers, follow the format and the conventions in
  `.claude/skills/references.rules.md`

// Ignore
//
// BOOK="book_AI_Software_Engineering"
// CHAP="## 14: Human-AI Collaboration Patterns"
//
// BOOK="book_Decentralized_AI"
// CHAP="## 01: Blockchain Consensus Foundations"
