# `affiliation_tiers.yaml`

- Dated list of the institutions that score 5 and 4 in the affiliation signal of the
  paper rating
- Read by `research/Estimate_academic_paper_quality/report_papers.py` in its `rate`
  stage
- Filled from the QS World University Rankings site with
  `research/Estimate_academic_paper_quality/populate_affiliation_tiers.py`, never
  from the memory of an LLM (decision D15 in
  `research/Estimate_academic_paper_quality/docs/03-decisions.md`)
- Empty until a person fills it: every affiliation then scores by its kind, e.g., a
  university scores 3

## What the File Contains

### Keys

| Key       | Type   | Meaning                                                           |
| :-------- | :----- | :---------------------------------------------------------------- |
| `as_of`   | string | Edition and access date of the source, e.g., `QS ... 2026, accessed 2026-11-03` |
| `source`  | string | URL of the page that the names and the ranks come from            |
| `tier_5`  | list   | Institutions that score 5                                         |
| `tier_4`  | list   | Institutions that score 4                                         |

### Entries

- **`name`** (required): full name of the institution, as a paper writes it
- **`aliases`** (optional): other names that a paper uses, e.g., an acronym or the
  name of a school of the institution
- **`qs_rank`** (optional): the QS rank. Only the script writes it. It marks the
  entry as owned by the script
- Any other key is ignored by `report_papers.py`, so a hand entry can carry a
  `note` or a `source`
- Example (the names are fixtures, not real ranks):
  ```yaml
  tier_5:
    - name: "Alpha University"
      aliases: ["Alpha", "Alpha Business School"]
      qs_rank: 1
    # Entries added by hand: the script keeps them.
    - name: "Test Research Lab"
      aliases: ["TRL"]
      note: "leading industry research lab"
  ```

### The Two Kinds of Entries

- **QS entries**: have a `qs_rank`. They come from the QS export. The script
  replaces them on each run
- **Hand entries**: have no `qs_rank`. A person adds them for what QS does not
  rank: industry research labs, NBER, and other research groups. The script keeps
  them as they are

### Tiers

| Tier     | Score | What goes in                                                       |
| :------- | :---- | :----------------------------------------------------------------- |
| `tier_5` | 5     | QS rank 1 to 20, leading industry research labs, NBER             |
| `tier_4` | 4     | QS rank 21 to 100, established industry research groups           |

- The rule is in `.claude/skills/paper.rules.md`, section "Affiliation"
- A central bank needs no entry: the kind `central_bank` that the LLM reads in the
  paper already scores 5 (`_KIND_SCORES` in `report_papers.py`)
- A lab and NBER need entries: their kinds (`company`, `research_institute`) score
  2 and 3

## How the Rater Uses the File

1. **Read**: the LLM reads the institution of each author from the paper, with its
   kind
2. **Normalize**: `report_papers.py` makes the name ASCII, lowercase, with no
   punctuation and no leading `the`
3. **Match**: an entry matches if its `name` or one `aliases` item is a whole-word
   part of the institution. The longest match wins. On a tie in length, the higher
   tier wins
4. **Score**: a match gives 5 or 4. No match gives the score of the kind: university
   3, research institute 3, company 2, none 1, central bank 5
5. **Pick**: the signal is the best score among the authors

## How to Fill the File

1. **Download the QS table** into the dir `research/Estimate_academic_paper_quality/`
   - Open https://www.topuniversities.com/world-university-rankings in a browser and
     pick the latest edition, e.g., 2027
   - Save the table as the CSV
     `research/Estimate_academic_paper_quality/qs_2027.csv`, with at least a rank
     column and an institution name column. The rank may be `=20` for a tie
   - If the page has no CSV export, use the official results table of the edition on
     `qs.com`, e.g., "QS World University Rankings 2027 Results Table (Excel)". QS
     emails the Excel file after a short form. Open it and save it as CSV
   - Write down the edition and the URL of the page for the next step
   - A script cannot do this step: the QS site answers `403` with a Cloudflare
     challenge (`cf-mitigated: challenge`) to `curl` and to a fetch tool, checked on
     2026-11-04. Do not work around the challenge
   - Layout of `research/Estimate_academic_paper_quality/qs_2027.csv` (checked on
     2026-11-04): 1504 rows, with the columns `Index`, `Rank`, `Previous Rank`,
     `Name`, `Country/Territory`, and others. The script finds `Rank` and `Name`
     with no flag. For another layout, pass `--rank_col` and `--name_col`
   - Check with `git status` if the CSV should be committed, since the QS terms of
     use may limit sharing the table

2. **Run the script**
   ```bash
   > research/Estimate_academic_paper_quality/populate_affiliation_tiers.py \
       --qs_file research/Estimate_academic_paper_quality/qs_2027.csv \
       --edition "QS World University Rankings 2027" \
       --source "https://www.topuniversities.com/world-university-rankings"
   ```
   - Add `--dry_run` to see the file without writing it
   - The script sets `as_of` to the edition plus the access date, and writes
     `tier_5` and `tier_4`

3. **Add aliases by hand**
   - The QS name is the formal name. A paper may write a school, a short name, or an
     acronym
   - Add them to `aliases`. The script keeps them on the next run
   - The script adds an acronym in parentheses as an alias, e.g.,
     `Test Institute of Technology (TIT)` gives the name
     `Test Institute of Technology` and the alias `TIT`
   - Avoid an alias that is also a part of another institution:
     - **Bad**: `Columbia` also matches `University of British Columbia`
     - **Bad**: `Cambridge` and `Washington` match many names
     - **Good**: `Columbia Business School`, `Columbia Law School`, or
       `Columbia University`

4. **Add the hand entries**
   - Add the industry labs and NBER to `tier_5` or `tier_4`, with no `qs_rank`
   - Rater open question 1 in
     `research/Estimate_academic_paper_quality/docs/02.6-design.rater.md` is still
     open: it asks which groups count as "established industry research group"
     (score 4)
   - Take the names from a source that you can cite, and say which source in a
     `note` key. Do not take them from the memory of an LLM

5. **Review the change**
   ```bash
   > git diff research/Estimate_academic_paper_quality/affiliation_tiers.yaml
   ```

6. **Rate again**. The `rate` stage makes no LLM call:
   ```bash
   > research/Estimate_academic_paper_quality/report_papers.py \
       --output_dir ~/papers/run_2026_10
   ```

## Check the Result

- List the institutions that the LLM read in a run, then look for a name that
  should be in a tier but scores as a plain university:
  ```bash
  > grep -rhoE '"institution": *"[^"]+"' ~/papers/run_2026_10 --include='*.features.json' \
      | sort | uniq -c | sort -rn
  ```
- Compare with the column `affiliation_score` of `04-report.report.json`: a 3 for a
  top 100 university means a missing alias
- Typical misses: a business school, a law school, or an institute of a university,
  since the name of the school does not contain the name of the university

## Refresh Each Year

- QS publishes a new edition each year
- Download the new CSV (step 1 of "How to Fill the File") and run the script again
  with the new `--edition` and `--source`
- The script keeps the aliases and the hand entries. It drops a QS entry that left
  the top 100 and moves an entry whose rank crossed the limit of 20
- Rate again, since the scores can change

## `populate_affiliation_tiers.py`

### Arguments

| Argument           | Type | Default                      | Description                                           |
| :----------------- | :--- | :--------------------------- | :---------------------------------------------------- |
| `--qs_file`        | str  | required                     | CSV exported from the QS site                         |
| `--edition`        | str  | required                     | Edition, e.g., `QS World University Rankings 2026`    |
| `--source`         | str  | required                     | URL of the QS page                                    |
| `--accessed`       | str  | today                        | Export date, `YYYY-MM-DD`                             |
| `--yaml_file`      | str  | `affiliation_tiers.yaml`     | File to write, found in the Git tree                  |
| `--rank_col`       | str  | found                        | Header of the rank column                             |
| `--name_col`       | str  | found                        | Header of the name column                             |
| `--tier_5_max_rank` | int | 20                           | Last QS rank that scores 5                            |
| `--tier_4_max_rank` | int | 100                          | Last QS rank that scores 4                            |
| `--dry_run`        | flag | false                        | Log the new file and do not write it                  |

### Behavior

- **Columns**: finds the rank column among `rank`, `rank_display`, `qs rank`, and
  `overall rank`, or the only header that contains `rank`. Finds the name column
  among `institution`, `institution name`, `university`, `name`, and `title`. It
  stops with an error if it cannot find one, and asks for the flag
- **Ranks**: `=20` is rank 20, `101-110` is rank 101, and `601+` is rank 601. A row
  with no number is skipped with a warning
- **Ties**: a tie at the last rank of a tier stays in that tier, so a tier can have
  more entries than its limit
- **Repeats**: a repeated name is dropped with a warning
- **Check**: the script warns if the CSV has fewer institutions of rank 1 to 20 than
  the limit, which means that the table is cut
- **Write**: it rewrites the whole file, with a new header comment. A comment that
  you wrote in the file is lost, so keep notes in a `note` key
- **Dependencies**: `pyyaml`, selected by the `uv run` line of the script

### Tests

- `research/Estimate_academic_paper_quality/test/test_populate_affiliation_tiers.py`
  checks the rank parsing, the column search, the merge with hand entries, and that
  `report_papers.py` reads the result and matches a name
- Run them from the repo root, with `PYTHONPATH` set to the repo root and
  `helpers_root`:
  ```bash
  > uv run --no-project --python 3.14 --with pytest --with pytest-timeout \
      --with pyyaml --with pydantic --with tqdm --with requests --with feedparser \
      --with beautifulsoup4 --with pymupdf --with openai --with pandas --with numpy \
      python -m pytest \
      research/Estimate_academic_paper_quality/test/test_populate_affiliation_tiers.py
  ```
