# How the Founder Age & Startup Success Dataset Was Generated

## Overview

This dataset supports the project A Causal Analysis of Age and Startup Success, which investigates whether founder age has a causal effect on startup outcomes, or whether the observed correlation is explained by confounding factors such as industry experience, network and financial capital, and team composition.

The dataset was built in two stages.

1. Company level merge, complete: combined Crunchbase and Y Combinator data on about 54,000 companies, with founder names, industry, funding, team size, and a derived success or failure outcome.
2. Founder age enrichment, LinkedIn URLs found and verified: sourced LinkedIn profile URLs for founders through the lab's PhantomBuster infrastructure, then cleaned and verified the results down to 1,500 confirmed founder to LinkedIn matches. Sourcing graduation years from those profiles, the remaining step needed to compute an age at founding proxy, is still pending.

## Data sources

Two raw exports were used, both loaded as is with no external enrichment at this stage.

| Source | File | Rows | Key fields |
| --- | --- | --- | --- |
| Crunchbase | investments\_VC.csv | about 54,300 companies | company name, market or industry, funding total, funding rounds, founded year, region, status |
| Y Combinator | yc\_companies.csv | about 5,000 companies | company name, industry, founded year, team size, active founders, batch, status |

Crunchbase is the larger, broader source and does not include founder names or team size. Y Combinator is smaller but includes founder names and team size, which Crunchbase lacks. Combining the two gives a fuller picture than either alone.

## Merge process

Companies were matched across the two sources by name, since neither dataset shares a common id.

1. Normalize both company name columns: lowercase, strip whitespace, remove punctuation.
2. Exact match on the normalized name first.
3. For names left unmatched, run a fuzzy match, using token sort ratio, against the Crunchbase name list, keeping matches at or above a 90 out of 100 similarity score.

Result of the merge:

| Match type | Companies |
| --- | --- |
| Exact match | 765 |
| Fuzzy match | 240 |
| Total matched, both sources | 1,005 |
| Crunchbase only, unmatched | about 53,300 |
| Total rows in final dataset | 54,318 |

Matched rows carry fields from both sources, for example team size and founder names from Y Combinator alongside funding data from Crunchbase. Crunchbase only rows keep company level fields but have no team size or founder name data, since Crunchbase does not track those.

## Outcome variable

Each source reports company status in its own wording, so the raw status was mapped to one binary success field.

| Source | Status value | Mapped outcome |
| --- | --- | --- |
| Crunchbase | acquired | success, 1 |
| Crunchbase | closed | failure, 0 |
| Crunchbase | operating | censored, left blank |
| Y Combinator | acquired, public | success, 1 |
| Y Combinator | inactive | failure, 0 |
| Y Combinator | active | censored, left blank |

Still operating companies are treated as censored rather than as failures, since the outcome is not yet resolved. Across the full 54,318 row dataset, 3,873 rows resolved to success, 2,810 resolved to failure, and 47,635 remain censored.

## Founder name enrichment

Of the 964 matched companies with a founder names field from Y Combinator, 251 had that field blank in the raw data. Two checks were run before treating these as genuinely missing.

1. Checked for a second entry of the same company elsewhere in the raw Y Combinator file, for example a different batch, that might carry founder names. None of the 251 had one.
2. Checked whether the long description or website fields mentioned founders. Zero descriptions mentioned a founder, and only 11 of 251 companies even had a website link on file.

Given that, founder names for the missing 251 were sourced manually and web searched one by one, matched back by company name. All 251 are now filled, verified by spot checking a sample against independent sources such as Crunchbase, TechCrunch, and company blog posts. The founder names field also had formatting issues in the raw text, quoted nicknames such as the word Elle inside quotation marks, and the word and used instead of a comma between co-founder names. Both were cleaned so every row uses a consistent comma separated list.

## Founder age enrichment, LinkedIn URLs found and verified

Neither source dataset includes founder birthdate or age. The plan is to estimate age at founding using an education end date proxy, age equals founding year minus graduation year plus about 22, sourced through the research group PhantomBuster based LinkedIn outreach infrastructure.

The founder list was reshaped from 964 companies, one row per company with a comma separated founder list, into 1,738 rows, one row per individual founder, since the lookup tool needs one name per row rather than a combined list. The reshaped file uses a column named fullName, matching what the LinkedIn URL Finder phantom expects as input, plus company name carried along on each row so results can be re-joined back afterward.

Planned two step pipeline:

1. LinkedIn URL Finder phantom: input, a Google Sheet of founder full names, output, a matched LinkedIn profile URL per name.
2. LinkedIn Profile Scraper phantom: input, those found URLs, output, full profile data per person, expected to include education history.

Both steps require a logged in LinkedIn session, so a lab collaborator runs the phantoms rather than running them directly. Still to confirm before the full run: whether the Profile Scraper output actually includes education or graduation year fields, and whether to run the full 1,738 founders or a smaller sample first, given a stated PhantomBuster throughput of roughly 150 profiles per day.

### Result: URLs found and dataset cleaned

The LinkedIn URL Finder phantom, run through the lab's PhantomBuster infrastructure, returned a candidate LinkedIn URL for each of the 1,738 rows in the reshaped founder list. Those results were then reviewed and cleaned, with Claude's help, down to a final file of 1,500 rows, each carrying a verified LinkedIn URL.

Rows were removed from the candidate list for three reasons.

- No source could be found to confirm the match, so the field was left blank. For example, a URL had been linked to a person named Stuart Ross at a company called Partnered with no verifiable source behind it, so that row was removed.
- The name belonged to someone who was not actually a company founder.
- The founder had no LinkedIn presence to find.

Of the 1,738 candidate rows, 238 were removed across these three reasons, leaving 1,500 verified founder to LinkedIn URL matches. Each retained row also carries a source note, for example a YC company page or a web search confirming the name and company together, and most rows carry a second check as well, for example confirming the URL is indexed under that person's name or that the live profile name matches.

Still pending: running the LinkedIn Profile Scraper phantom against these 1,500 verified URLs to pull profile data, including education history, in order to source graduation years for the age at founding proxy.

## Known limitations

- Founder age is a derived proxy from an education end date, not a verified birthdate, so it carries measurement error.
- Team size and founder count are only available for the 1,005 companies matched to Y Combinator, not the full 54,318 row dataset, so those fields should not be treated as representative of Crunchbase only companies.
- Name based matching across sources, and the manual founder name lookups, both carry some risk of mismatch, though a sample was spot checked against independent sources.
- This data gap is not unique to this project. Published research in this area faces the same constraint. Azoulay and colleagues used restricted access Census and IRS microdata not available outside government research partnerships. Roche, Conti, and Rothaermel supplemented Crunchbase with paid sources and manual LinkedIn lookups and never released the merged dataset. Ali Tamaseb, author of Super Founders, spent about four years manually collecting founder level data for around 500 companies. Crunchbase is the standard backbone dataset across this literature, and this project uses that same base source.
- The 1,500 verified LinkedIn matches were checked by source note and, for most rows, a second confirmation step, for example a live profile name match or the URL being indexed under that name, rather than by independently auditing every match, so a small number of mismatches could remain.
