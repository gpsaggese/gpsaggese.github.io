# Dataset: how it was gathered

Author: Shobha Gupta | Causal Analysis of Age and Startup Success

This section records every step used to build the founder dataset, what tool was used at each step, how many rows were kept or removed, and what is still unverified. Numbers come from the files in this folder as of November 12, 2026.

## Summary

| Stage | What was done | Result |
|---|---|---|
| 1 | Merged Crunchbase and Y Combinator (YC) company data | 54,318 companies, 1,005 matched to YC |
| 2 | Filled missing founder names by hand | 251 companies filled |
| 3 | Found and verified LinkedIn URLs for each founder | 1,500 verified founder rows (810 companies) |
| 4 | Added founding year, graduation year, published age, and education notes | Age at founding for 302 founder rows |
| 5 | Checked current status (acquired, public, inactive, active) per company | 964 companies |
| 6 | Added founders from a Kaggle YC dataset | 553 founder rows (305 companies) added to the main sheet |
| 7 | Merged, de-duplicated, and cleaned | 2,047 rows in the main sheet |

## Stage 1: Company-level merge (Crunchbase + Y Combinator)

- **Sources:** Crunchbase `investments_VC.csv` (about 54,300 companies) and a YC company file `yc_companies.csv` (about 5,000 companies). Crunchbase has funding, industry, region and status but no founder names or team size. YC has founder names, team size and batch.
- **Matching:** the two files share no id, so companies were matched by name. Names were lowercased and stripped of punctuation. Exact normalized matches were taken first (765 companies). Remaining names were matched with fuzzy matching (token sort ratio, cutoff 90 out of 100, 240 companies). Total matched: 1,005.
- **Tools:** Python, pandas, rapidfuzz.
- **Outcome field:** each source words status differently, so it was mapped to a binary success field. Acquired or public = 1. Closed or inactive = 0. Operating or active = left blank (censored, because the outcome is not decided yet). Across the 54,318 rows: 3,873 success, 2,810 failure, 47,635 censored.
- **Note:** 964 matched companies were carried into the founder steps below. The Kaggle step (Stage 6) later added 305 companies, so the final main sheet covers 1,115 companies (810 original + 305 Kaggle).

## Stage 2: Founder names

- 251 of the 964 companies had a blank founder-names field in the YC file.
- Two checks before treating them as truly missing: (1) looked for a second entry of the same company elsewhere in the YC file (none of the 251 had one); (2) checked whether descriptions or website links named founders (zero descriptions did, and only 11 of 251 had a website).
- Founder names for all 251 were then found manually by web search, one company at a time, and matched back by company name. A sample was spot-checked against independent sources (Crunchbase, TechCrunch, company blog posts).
- Formatting cleanup: quoted nicknames were removed and the word "and" between co-founders was replaced by a comma, so every row is a consistent comma-separated list.
- Files: `companies_missing_founder_names.csv`, `companies_missing_founders_with_websites.csv`, `matched_founders_list.csv` (964 companies), `matched_founders_list_cleaned.xlsx`.

## Stage 3: LinkedIn URLs

- The 964 companies were reshaped from one row per company to one row per founder: 1,738 rows with a `fullName` column, matching what the LinkedIn URL Finder expects. Files: `founders_long_for_phantombuster.csv`, `founders_for_linkedin_url_finder.csv`, `founders_ready_for_linkedin_lookup.csv`, and a 30-row test file `founders_sample_30_for_testing.csv`.
- **Tool:** the PhantomBuster LinkedIn URL Finder, run through the research group's account by a lab collaborator, because it needs a logged-in LinkedIn session. Input: a Google Sheet of founder names. Output: a candidate profile URL per name.
- **Cleaning and verification (with Claude's help):** 238 of the 1,738 candidate rows were removed, leaving 1,500 verified rows. Reasons: no source could confirm the match; the person was not actually a founder; or the founder had no LinkedIn presence. The removed rows are kept in the `Removed Rows` tab. That tab has 234 rows.
- **Source note per row** (`LIN_url_source`): YC company page, or web search confirming name and company together.
- **Second check per row** (`link_check`): URL confirmed as indexed under that person's name (803 rows), live profile name matches (238 rows), or matched via a search result (12 rows). 107 rows say "Couldn't confirm - open manually" and are listed under open items.
- **Result:** 1,500 founder rows across 810 companies. Founders with no URL are kept on a separate tab (`Founders - No LinkedIn Found`).

## Stage 4: Founder age at founding

Neither Crunchbase nor YC has birthdates, so age is a proxy.

| Column | Meaning |
|---|---|
| Verified Founding Year | Founding year checked against sources (548 rows) |
| Bachelor's Grad Year | Graduation year found for the founder (230 rows) |
| Published Age, Year Age Published | An age printed in a source, and the year of that source (110 rows) |
| Age at Founding | Published age adjusted to the founding year; otherwise founding year minus bachelor's graduation year plus 22 |
| Age Method | How the age was obtained: exact (published age or birth date) or estimated (grad year, inferred from school year); "Not found" for 359 rows |
| Basis | Free-text note with the source and education details |

- 661 rows have an Age Method entry; 302 of those have an age and 359 are "Not found".
- The assumed age of 22 at a bachelor's graduation is kept in one cell so it can be changed.
- Education flags used in the models (Ivy League, master's, PhD, dropout) are read from the Basis text with keyword matching, and the matches were audited by hand.
- Grad years and published ages come from public sources recorded in the Basis column. The PhantomBuster Profile Scraper step has not been run on the new Kaggle founders.

## Stage 5: Company status

- For each of the 964 companies, current status was checked and recorded (`company_current_status.xlsx`): 268 acquired, 275 active, 269 inactive, 13 public, 139 unclear.
- Sources recorded per company: YC company page (668), web search (270), and general or background knowledge (26).
- The status was merged into the founders sheet by company name (`current_status`, `status_source`, `status_notes`).
- **Success** in the models is acquired or public. **Failure** is inactive. Active and unclear are excluded because the outcome is not decided.

## Stage 6: Kaggle YC dataset

- **Why:** to grow the sample. The Kaggle file has 1,000 YC companies and 2,047 founder rows, with LinkedIn links for 1,565 founders, but no founder age, no education, and no failed companies (status tags: 956 Active, 29 Acquired, 15 Public, 0 Inactive).
- **Overlap:** 175 of our 810 companies were already in it. 821 companies were new, of which 429 were from the 2022 batch (too recent for outcomes) and were dropped. 376 companies from the 2012 to 2021 batches were kept, with 757 founders (558 with a LinkedIn URL, 199 without).
- **Status check for the 376:** done in parallel by eight web-search workers (Claude), each reading the company's YC page and running a shutdown or acquisition search. Result: 277 active, 58 acquired, 21 inactive, 7 public, 13 unclear. Each company also has a one-line description of what it does.
- **Confidence column (`status_confidence`):** low (209) means active on the YC badge only, with no news check. Medium (48) is a non-active badge with no second source. Higher (106) is confirmed by news or the company's own site. None (13) is unclear.
- **Why the Kaggle tags were not used as outcomes:** they disagree with our own checks on about 30 of the 175 shared companies (for example, 15 we found acquired and 15 we found inactive that Kaggle shows as Active). The YC badge itself is also unreliable (for example Flirtey, Routable, Convictional).
- Founding year for these rows is from Kaggle or the batch year and is **not verified**. Verified Founding Year, grad year, published age, and Basis are blank until the age step is run on them.

## Stage 7: Merge and de-duplication

- The 558 Kaggle founders with a URL were added to the main sheet in the same column layout (553 remained after removing duplicates of companies already in the sheet). The 199 without a URL went to the no-LinkedIn tab.
- Duplicate check by company, founder and LinkedIn URL:
  - 4 Kaggle companies were duplicates of companies already in the sheet under a different spelling (for example Theorem LP vs Theorem, Quit Genius vs Pelago) and were removed.
  - 6 repeated rows in the original 1,500 were removed (kept the first of each pair; one pair had conflicting founding years, 2009 vs 2014, and the 2014 row was kept), leaving 1,494.
  - 3 rows had a LinkedIn URL that belonged to a different person. The URL was cleared and the row moved to the no-LinkedIn tab.
  - Same URL on several rows is allowed only when it is the same founder at several companies (serial founders).
- The age-at-founding formula was rewritten for every row and the cross-references on the no-LinkedIn tab were recomputed.
- **Final main sheet:** 2,047 rows = 1,494 original + 553 Kaggle. No-LinkedIn tab: 261 founders. Removed Rows tab: 234 rows.

| Source | Rows | Companies | Acquired | Public | Inactive | Active | Unclear |
|---|---|---|---|---|---|---|---|
| Original | 1,494 | 810 | 495 | 30 | 434 | 465 | 70 |
| Kaggle | 553 | 305 | 100 | 13 | 23 | 398 | 19 |

## How the data is used in the analysis

- Age, Ivy League, and degree analyses use only rows with age or education data, which are all original rows.
- The Kaggle rows have a batch and a status but no age or education, so they only affect the batch analysis. They show 83% success against 54% for the original rows, because the Kaggle file keeps surviving companies and has almost no failures. Batch results are therefore reported on the original rows only (`USE_SOURCES = ("original",)` in `02_modeling_v3.ipynb`), and the pooled version is shown as a caveat.
- See `modeling_comparison.md` for the first run against the rerun.

## Tools used

| Tool | Used for |
|---|---|
| Kaggle | Crunchbase export, YC company file, YC directory dataset |
| Python (pandas, rapidfuzz) | Cleaning, name matching, merging |
| PhantomBuster (LinkedIn URL Finder) | Finding LinkedIn URLs, run by a lab collaborator |
| Google Sheets / Excel (openpyxl) | Working sheets, formulas, tabs |
| Claude | Cleaning and verifying URLs, web searches for status and founder names, one-line company descriptions, notebooks |
| Web search and company pages | Manual founder-name lookups, status checks, age sources |
| Jupyter, statsmodels, scipy, matplotlib | Analysis in `01_eda.ipynb` and `02_modeling_v3.ipynb` |
| GitHub | Version control and pull requests |

## Sources considered and not used

Wellfound (terms prohibit automated access, no age), Crunchbase API (paid), Crustdata student program (fields and eligibility unconfirmed), and restricted sources (Census LBD, IRS K-1, VentureXpert). Open options for later: SEC Item 401 (public companies only), Wikidata (low coverage, untested), Kauffman Firm Survey and PSED II (different populations). Details and links are in the dataset doc.

## Known limitations

- Age is a derived proxy, not a verified birthdate, and is only known for about 300 founders, mostly those with a public profile. That may favor more successful companies.
- Name-based matching and manual founder-name lookups can introduce mismatches. A sample was spot-checked, not every row.
- Most Kaggle statuses are unverified (209 of 376 rest on the YC badge alone), and a missed shutdown leaves a company marked active, so failures look rarer than they are.
- Founding year for Kaggle rows is unverified.
- Some statuses (26) come from general knowledge, not a source.

## Open items

- Run the Profile Scraper on the 553 new founders to add age and education.
- Second source check on the 209 low-confidence and 13 unclear Kaggle statuses.
- Resolve the 107 LinkedIn links marked "Couldn't confirm".
- Re-check the 26 statuses taken from general knowledge.
- Then propensity score matching with sensitivity analysis.

## Files

- Datasets: `founders_linkedin_filled_final_with_kaggle.xlsx` (main merged sheet), `founders_linkedin_filled_final.xlsx` (before the Kaggle merge), `company_current_status.xlsx`, `kaggle_new_founders_with_status.xlsx`, `kaggle_new_founders_matched_format.xlsx`.
- Intermediate: the CSVs listed under Stages 2 and 3.
- Code: `data_collection.py`, `01_eda.ipynb`, `02_modeling.ipynb` (first run), `02_modeling_v3.ipynb` (rerun).
