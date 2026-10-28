# Estimate Paper Quality Product Requirements

## Goal

- Build a pipeline that turns a loose list of papers into a CSV table of candidate
  trading strategies, with the full text and a set of extracted features per paper
- Users: quant researchers, who use the table to pick the next papers to replicate
- Background:
  - Idea: `research/Estimate_academic_paper_quality/idea.md`
  - Reference format: `.claude/skills/references.rules.md`
  - Paper rating rules, applied for Paper quality: `.claude/skills/paper.rules.md`,
    `# Paper Rating`
  - Download and conversion tool:
    `helpers_root/dev_scripts_helpers/download/download_academic_paper_to_md.py`

## Users and Use Cases

- **Quant researcher, first run**:
  - Pastes a list of papers collected from many sources (partial titles, URLs, DOIs)
    and gets one clean list of references
  - Gives a local PDF for a paper that the pipeline cannot download
  - Gets the full text of each paper as markdown, to read it and to feed it to other
    tools
  - Compares the papers feature by feature (e.g., reported Sharpe ratio, asset
    class), to judge how promising they are and how easy they are to replicate
  - Reads where each feature value comes from, to check it in the paper
- **Quant researcher, rerun**:
  - Adds new papers to a list that was already processed
  - Gets an updated CSV without paying again for the papers already processed

## Requirements

### Input

- `R1`: The pipeline accepts a text file with one paper per line, where a line is any
  of:
  - A partial or full title, optionally followed by the authors in parentheses
  - An arXiv URL or arXiv ID
  - A DOI or DOI URL
  - An SSRN URL
  - A publisher URL or a direct PDF URL
  - A path to a local PDF, e.g., for a paper behind a paywall or an SSRN block

### Normalize the List

- `R2`: For each resolved line, the pipeline outputs one reference in the format of
  `.claude/skills/references.rules.md`, `# Format for General Text`
  - E.g., `2009, Pearl, "Causality: Models, Reasoning, and Inference" (<Link>)`
- `R3`: The year, the first author, and the title of each reference match a
  bibliographic source (arXiv, Crossref, or Semantic Scholar), not the memory of an
  LLM
- `R4`: Each reference has a download link to the full text, or a mark that no
  download link was found
- `R5`: When a line matches no paper or more than one paper, the pipeline:
  - Marks the line as unresolved
  - Lists the candidate papers, if any
  - Goes on with the other lines
- `R6`: Two input lines that resolve to the same paper produce one reference
- `R7`: The normalized list is in reverse chronological order, per
  `.claude/skills/references.rules.md`

### Download and Convert

- `R8`: For each resolved paper with a download link or with a local PDF, the
  pipeline saves the PDF and its markdown conversion
  - The PDFs are saved in the repo
- `R9`: Both files use the base name of `.claude/skills/references.rules.md`, `#
  Format to Use in File Names`
  - E.g., `2016.Ribeiro_et_al.Why_Should_I_Trust_You.pdf` and `.md`
- `R10`: When the download or the conversion of a paper fails, the pipeline records
  the reason for that paper and goes on with the other papers
- `R11`: A rerun does not download or convert again a paper whose PDF and markdown
  already exist

### Extract Features

- `R12`: The pipeline reports each converted paper with these features:

  | Feature            | What it reports                                              |
  | :----------------- | :----------------------------------------------------------- |
  | Paper quality      | Rating from 1 to 5 of page length, affiliation, and citations (see `R16`) |
  | Performance        | Reported Sharpe ratio and other performance metrics, as free text |
  | Recency            | Publication year                                             |
  | Asset class        | Target asset class, e.g., US equities, as free text          |
  | Theoretical domain | Domain of the theory, e.g., micro vs. macro, as free text    |
  | Mechanism          | Nature of the reported mechanism: causal vs. associational, as free text |

- `R13`: Each feature has a value, or `unknown` when the paper and the lookup sources
  do not give the information
  - The pipeline does not convert a feature into a score, except the Paper quality
    rating of `R16`
  - E.g., a paper that describes no trading strategy (a survey, a method paper) has
    `unknown` for Performance, and stays in the report
- `R14`: Each feature value comes with its evidence:
  - The text that it is based on, e.g., `Sharpe 1.4, out of sample, 2005 to 2020`
  - Where the text comes from: section, table, or page of the paper, or the name of
    the external source
- `R15`: The pipeline extracts these facts from each paper and reports them as free
  text, as the paper states them:
  - Reported performance metrics: Sharpe ratio, annual return, volatility, max
    drawdown
    - When the paper gives several Sharpe ratios (e.g., gross and net of costs, in
      sample and out of sample), the pipeline reports all of them with the label
      that the paper gives
  - Sample period, and in sample vs. out of sample
  - Target asset class and universe
  - Data the paper needs: assets, frequency, fields
- `R16`: The Paper quality rating follows `.claude/skills/paper.rules.md`,
  `# Paper Rating`:
  - Three signals: page length, best affiliation of the authors, and citation count
  - Each signal scores from 1 to 5, and the rating is the mean of the known signals,
    rounded half up to a whole number
  - A signal that the pipeline does not find is `unknown` and is left out of the mean
  - The report gives the rating, the score of each signal, and the source and the
    date of the citation count
- `R17`: A rerun does not extract again the features of a paper that already has
  them, unless the user asks for it

### Output

- `R18`: The pipeline writes one CSV report with one row per paper, in the order of
  `R7`, with these columns:
  - The reference
  - The value of each feature
  - The score of each signal of the Paper quality rating
- `R19`: The pipeline lists the unresolved papers and the failed papers, with the
  reason, in a CSV apart from the report of `R18`
- `R20`: For each converted paper, the pipeline writes a features file with the
  feature values, the evidence, and the extracted facts
  - The features file uses the same base name as the PDF of the paper (see `R9`)

## Non-Goals

- Does not search for papers: it works only on the list from the user
- Does not replicate or backtest the strategy of a paper
- Does not check that the reported results are true: it reports what the paper claims
- Does not bypass paywalls or access controls: a paper without a public PDF is
  reported as failed, unless the user gives a local PDF
  - SSRN papers are in scope: a public SSRN PDF is not behind an access control, so
    the download tool gets it through a Chrome that it starts and closes
- Does not combine the features into a total score and does not rank the papers: the
  user sorts the CSV
- Does not replace `.claude/skills/paper.rate/SKILL.md`, which rates one paper: the
  pipeline applies the same rules to a list (see `R16`)

## Success Metrics

- **Resolution accuracy**: at least 95% of the lines of a hand-labeled input list
  resolve to the right paper (first author, year, and title match)
- **Download coverage**: at least 80% of the resolved papers that have a public PDF
  yield both a PDF and a markdown file
- **Extraction accuracy**: at least 90% of the extracted Sharpe ratios match the
  value that a researcher reads in the paper, on a labeled set of 20 papers

## Constraints and Assumptions

- Download and conversion reuse
  `helpers_root/dev_scripts_helpers/download/download_academic_paper_to_md.py`, as
  the idea states
- Bibliographic facts and citation counts come from a lookup source, never from the
  memory of an LLM, per `.claude/skills/paper.rules.md`
- An input list has about 100 papers: a first run processes all of them, and a rerun
  does not pay again for the papers already processed (see `R11` and `R17`)
- Most papers describe trading strategies or return predictors, but a paper may not
  (e.g., a survey, a method paper)
  - The features Performance and Asset class can be `unknown` for such a paper (see
    `R13`)
- SSRN blocks scripted downloads: the download tool gets SSRN PDFs through a Chrome
  that it starts and closes (`download_with_chrome.py`), so the pipeline needs
  Chrome
- The user runs the pipeline with network access

## Open Questions

None.
